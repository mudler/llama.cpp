#include "moe-ffn.cuh"
#include "getrows.cuh"
#include "mmq.cuh"
#include "unary.cuh"

#include <cstdlib>

bool ggml_cuda_moe_routed_ffn_poc_enabled() {
    static const bool enabled = [] {
        const char * value = getenv("LLAMA_MOE_ROUTED_FFN_POC");
        return value != nullptr && atoi(value) != 0;
    }();
    return enabled;
}

bool ggml_cuda_moe_routed_ffn_poc_should_engage(
        const ggml_tensor * gate_up,
        const ggml_tensor * gate,
        const ggml_tensor * up,
        const ggml_tensor * glu,
        const ggml_tensor * down,
        const ggml_tensor * ids,
        int cc) {
    if (!blackwell_mma_available(cc)) {
        return false;
    }
    if (gate_up == nullptr || gate == nullptr || up == nullptr || glu == nullptr || down == nullptr || ids == nullptr) {
        return false;
    }
    if (gate_up->op != GGML_OP_MUL_MAT_ID || down->op != GGML_OP_MUL_MAT_ID) {
        return false;
    }
    if (gate->op != GGML_OP_VIEW || up->op != GGML_OP_VIEW || gate->view_src != gate_up || up->view_src != gate_up) {
        return false;
    }
    if (glu->op != GGML_OP_GLU || ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU || down->src[1] != glu) {
        return false;
    }
    if (gate_up->src[2] != ids || down->src[2] != ids) {
        return false;
    }

    const ggml_tensor * down_w = down->src[0];
    if (down_w == nullptr || (down_w->type != GGML_TYPE_NVFP4 && down_w->type != GGML_TYPE_MXFP4)) {
        return false;
    }

    return true;
}

static bool ggml_cuda_moe_routed_ffn_fused_quant_enabled() {
    static const bool enabled = [] {
        const char * value = getenv("LLAMA_MOE_ROUTED_FFN_FUSED_QUANT");
        return value != nullptr && atoi(value) != 0;
    }();
    return enabled;
}

static bool ggml_cuda_moe_routed_ffn_down_supported(const ggml_tensor * glu, const ggml_tensor * down) {
    const ggml_tensor * down_w = down != nullptr ? down->src[0] : nullptr;
    const ggml_tensor * ids    = down != nullptr ? down->src[2] : nullptr;
    if (glu == nullptr || down == nullptr || down_w == nullptr || ids == nullptr) {
        return false;
    }
    if (down_w->type != GGML_TYPE_NVFP4 && down_w->type != GGML_TYPE_MXFP4) {
        return false;
    }
    if (glu->type != GGML_TYPE_F32 || down->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32) {
        return false;
    }
    if (glu->ne[3] != 1 || down->ne[3] != 1 || ids->ne[2] != 1 || ids->ne[3] != 1) {
        return false;
    }
    if (down_w->ne[0] != glu->ne[0] || down_w->ne[1] != down->ne[0] || down_w->ne[2] <= 0) {
        return false;
    }
    if (ids->ne[0] != glu->ne[1] || ids->ne[1] != glu->ne[2]) {
        return false;
    }
    if (down->ne[1] != glu->ne[1] || down->ne[2] != glu->ne[2]) {
        return false;
    }
    if (ids->nb[0] != ggml_element_size(ids)) {
        return false;
    }
    if (glu->nb[0] != sizeof(float) || down->nb[0] != sizeof(float)) {
        return false;
    }
    if (glu->nb[1] != (size_t) (glu->ne[0] * (int64_t) sizeof(float)) ||
        glu->nb[2] != (size_t) (glu->ne[1] * (int64_t) glu->nb[1])) {
        return false;
    }
    if (down->nb[1] != (size_t) (down->ne[0] * (int64_t) sizeof(float)) ||
        down->nb[2] != (size_t) (down->ne[1] * (int64_t) down->nb[1])) {
        return false;
    }

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    return ggml_cuda_should_use_mmq(down_w->type, cc, glu->ne[2], down_w->ne[2]);
}

static __global__ void moe_swiglu_nvfp4_quant_kernel(
        const float * __restrict__ gate,
        const float * __restrict__ up,
        const int32_t * __restrict__ ids_src1,
        void * __restrict__ vy,
        int64_t n_ff,
        int64_t n_ff_padded,
        int64_t n_rows,
        int64_t n_used,
        int64_t gate_s1,
        int64_t gate_s2,
        int64_t up_s1,
        int64_t up_s2) {
#if defined(BLACKWELL_MMA_AVAILABLE)
    const int64_t i0_base = ((int64_t) blockDim.x * blockIdx.y + threadIdx.x) * QK_NVFP4_SUB;
    if (i0_base >= n_ff_padded) {
        return;
    }

    const int64_t row = blockIdx.x;
    const int64_t src_row = ids_src1[row];
    const int64_t token = src_row / n_used;
    const int64_t used = src_row - token * n_used;
    const int64_t k_block = i0_base / QK_K;
    const int64_t blocks_per_col = (n_ff_padded + QK_K - 1) / QK_K;
    if (k_block >= blocks_per_col) {
        return;
    }

    block_fp4_mmq * y = (block_fp4_mmq *) vy;
    block_fp4_mmq * yb = y + k_block * n_rows + row;
    const int sub = (i0_base % QK_K) / QK_NVFP4_SUB;

    float vals_raw[QK_NVFP4_SUB];
    float amax_raw = 0.0f;
#pragma unroll
    for (int k = 0; k < QK_NVFP4_SUB; k++) {
        const int64_t i0 = i0_base + k;
        if (i0 < n_ff) {
            const float g = gate[token * gate_s2 + used * gate_s1 + i0];
            const float u = up[token * up_s2 + used * up_s1 + i0];
            const float v = ggml_cuda_op_silu_single(g) * u;
            vals_raw[k] = v;
            amax_raw = fmaxf(amax_raw, fabsf(v));
        } else {
            vals_raw[k] = 0.0f;
        }
    }

    static constexpr int test_offsets[5] = { 0, -1, 1, -2, 2 };
    const int first_fp8_code = (int) ggml_cuda_fp32_to_ue4m3(amax_raw / 6.0f);

    float best_err = FLT_MAX;
    uint8_t fp8_code = 0;
    float subblock_scale = 0.0f;

#pragma unroll
    for (int i = 0; i < 5; i++) {
        const int test_code = first_fp8_code + test_offsets[i];
        if (test_code < 0 || test_code > 0x7e) {
            continue;
        }
        const uint8_t code = (uint8_t) test_code;
        const float test_scale = ggml_cuda_ue4m3_to_fp32(code);
        const float test_inv_scale = test_scale > 0.0f ? 0.5f / test_scale : 0.0f;
        float cur_err = 0.0f;
#pragma unroll
        for (int k = 0; k < QK_NVFP4_SUB; ++k) {
            const float v = vals_raw[k];
            const uint8_t q = ggml_cuda_float_to_fp4_e2m1(v, test_inv_scale);
            const float err_diff = fabsf(v) - fabsf(kvalues_mxfp4[q & 0x7]) * test_scale;
            cur_err = fmaf(err_diff, err_diff, cur_err);
        }

        if (cur_err < best_err) {
            best_err = cur_err;
            fp8_code = test_code;
            subblock_scale = test_scale;
        }
    }

    const float inv_scale = subblock_scale > 0.0f ? 0.5f / subblock_scale : 0.0f;
    uint32_t q0 = 0;
    uint32_t q1 = 0;
#pragma unroll
    for (int k = 0; k < QK_NVFP4_SUB / 4; ++k) {
        q0 |= (uint32_t) ggml_cuda_float_to_fp4_e2m1(vals_raw[k +  0], inv_scale) << (8 * k);
        q0 |= (uint32_t) ggml_cuda_float_to_fp4_e2m1(vals_raw[k +  8], inv_scale) << (8 * k + 4);
        q1 |= (uint32_t) ggml_cuda_float_to_fp4_e2m1(vals_raw[k +  4], inv_scale) << (8 * k);
        q1 |= (uint32_t) ggml_cuda_float_to_fp4_e2m1(vals_raw[k + 12], inv_scale) << (8 * k + 4);
    }

    uint32_t * yqs = reinterpret_cast<uint32_t *>(yb->qs);
    yqs[2 * sub + 0] = q0;
    yqs[2 * sub + 1] = q1;
    reinterpret_cast<uint8_t *>(yb->d4)[sub] = fp8_code;
#else
    NO_DEVICE_CODE;
#endif
}

static bool ggml_cuda_moe_routed_ffn_fused_quant(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * gate,
        ggml_tensor * up,
        ggml_tensor * glu,
        ggml_tensor * down) {
    if (!ggml_cuda_moe_routed_ffn_fused_quant_enabled()) {
        return false;
    }
    if (!ggml_cuda_moe_routed_ffn_down_supported(glu, down)) {
        return false;
    }
    const ggml_tensor * down_w = down->src[0];
    const ggml_tensor * ids = down->src[2];
    if (down_w->type != GGML_TYPE_NVFP4) {
        return false;
    }
    if (gate == nullptr || up == nullptr || gate->type != GGML_TYPE_F32 || up->type != GGML_TYPE_F32) {
        return false;
    }
    if (gate->ne[0] != glu->ne[0] || gate->ne[1] != glu->ne[1] || gate->ne[2] != glu->ne[2] || gate->ne[3] != glu->ne[3]) {
        return false;
    }
    if (up->ne[0] != glu->ne[0] || up->ne[1] != glu->ne[1] || up->ne[2] != glu->ne[2] || up->ne[3] != glu->ne[3]) {
        return false;
    }
    if (gate->nb[0] != sizeof(float) || up->nb[0] != sizeof(float)) {
        return false;
    }

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (!blackwell_mma_available(cc)) {
        return false;
    }

    const int64_t n_ff = glu->ne[0];
    const int64_t n_ff_padded = GGML_PAD(n_ff, MATRIX_ROW_PADDING);
    if (n_ff % QK_NVFP4 != 0) {
        return false;
    }

    const int64_t n_expert_used = ids->ne[0];
    const int64_t n_tokens = glu->ne[2];
    const int64_t n_experts = down_w->ne[2];
    const int64_t ne_get_rows = n_tokens * n_expert_used;

    ggml_cuda_mmq_ids_meta ids_meta(ctx.pool(), ne_get_rows, n_experts);
    const int64_t sis1 = glu->nb[2] / glu->nb[1];
    ids_meta.build(ids, n_experts, n_tokens, n_expert_used, glu->ne[1], sis1, ctx.stream());

    const size_t nbytes_src1_q = ne_get_rows * n_ff_padded * sizeof(block_fp4_mmq) / QK_K +
        get_mmq_x_max_host(cc) * sizeof(block_q8_1_mmq);
    ggml_cuda_pool_alloc<char> src1_q(ctx.pool(), nbytes_src1_q);

    constexpr int nvfp4_block_size = 128;
    const int64_t block_num_y = (n_ff_padded + QK_NVFP4_SUB * nvfp4_block_size - 1) / (QK_NVFP4_SUB * nvfp4_block_size);
    const dim3 block_size(nvfp4_block_size, 1, 1);
    const dim3 num_blocks(ne_get_rows, block_num_y, 1);
    moe_swiglu_nvfp4_quant_kernel<<<num_blocks, block_size, 0, ctx.stream()>>>(
        (const float *) gate->data, (const float *) up->data, ids_meta.ids_src1.get(), src1_q.get(),
        n_ff, n_ff_padded, ne_get_rows, n_expert_used,
        gate->nb[1] / sizeof(float), gate->nb[2] / sizeof(float),
        up->nb[1] / sizeof(float), up->nb[2] / sizeof(float));
    CUDA_CHECK(cudaGetLastError());

    ggml_cuda_mul_mat_q_moe_quantized(
        ctx, down_w, src1_q.get(), down,
        ids_meta.ids_dst.get(), ids_meta.expert_bounds.get(),
        n_tokens, n_expert_used, n_experts, n_ff_padded);
    return true;
}

bool ggml_cuda_moe_routed_ffn_poc(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * gate_up,
        ggml_tensor * gate,
        ggml_tensor * up,
        ggml_tensor * glu,
        ggml_tensor * down) {
    if (!ggml_cuda_compute_forward(ctx, gate_up)) {
        return false;
    }
    if (ggml_cuda_moe_routed_ffn_fused_quant(ctx, gate, up, glu, down)) {
        return true;
    }
    if (!ggml_cuda_compute_forward(ctx, glu)) {
        return false;
    }
    if (!ggml_cuda_compute_forward(ctx, down)) {
        return false;
    }

    return true;
}
