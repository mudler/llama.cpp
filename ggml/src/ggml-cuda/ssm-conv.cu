#include "common.cuh"
#include "ssm-conv.cuh"
#include "unary.cuh"

template <bool apply_silu, size_t split_d_inner, size_t d_conv>
static __global__ void ssm_conv_f32(const float * src0_ptr, const float * src1_ptr,
                                    const float * bias_ptr,
                                    const int src0_nb0, const int src0_nb1, const int src0_nb2, const int src1_nb1,
                                    float * dst_ptr, const int dst_nb0, const int dst_nb1, const int dst_nb2,
                                    const int64_t n_t) {
    ggml_cuda_pdl_lc();
    const float * GGML_CUDA_RESTRICT src0 = src0_ptr;
    const float * GGML_CUDA_RESTRICT src1 = src1_ptr;
    const float * GGML_CUDA_RESTRICT bias = bias_ptr;
    float       * GGML_CUDA_RESTRICT dst  = dst_ptr;
    GGML_UNUSED(src0_nb0);
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block = (float *) ((char *) dst + bidx * dst_nb2 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    float x[d_conv] = { 0.0f };
    float w[d_conv] = { 0.0f };

    ggml_cuda_pdl_sync();
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    for (int64_t i = 0; i < n_t; i++) {
        float sumf = 0.0f;

        if (i == 0) {
            for (size_t j = 0; j < d_conv; j++) {
                x[j] = x_block[tid * stride_x + j];
            }
        } else {
            x[(i - 1) % d_conv] = x_block[tid * stride_x + i + d_conv - 1];
        }

#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += x[(i + j) % d_conv] * w[j];
        }
        sumf += b;
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

template <bool apply_silu, size_t split_d_inner, size_t d_conv, int64_t split_n_t>
static __global__ void ssm_conv_long_token_f32(const float * __restrict__ src0, const float * __restrict__ src1,
                                               const float * __restrict__ bias,
                                               const int src0_nb0, const int src0_nb1, const int src0_nb2,
                                               const int src1_nb1, float * __restrict__ dst, const int dst_nb0,
                                               const int dst_nb1, const int dst_nb2, const int64_t n_t) {
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;
    const int bidz = blockIdx.z;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1 +
                                             bidz * split_n_t * src0_nb0);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block =
        (float *) ((char *) dst + bidx * dst_nb2 + bidz * split_n_t * dst_nb1 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    const int64_t local_n_t = min(split_n_t, n_t - bidz * split_n_t);
    const int     n_cols    = d_conv - 1 + split_n_t;

    extern __shared__ float smem[];

    constexpr int load_cols   = d_conv - 1 + split_n_t;
    constexpr int total_elems = split_d_inner * load_cols;
    int row = tid / load_cols;
    int col = tid % load_cols;
#pragma unroll
    for (int idx = 0; idx < total_elems; idx += split_d_inner) {
        if (row < (int)split_d_inner) {
            smem[row * n_cols + col] = x_block[row * stride_x + col];
        }

        col += split_d_inner;
        row += col / load_cols;
        col  = col % load_cols;
        if (idx >= total_elems - tid - split_d_inner) {
            break;
        }
    }
    __syncthreads();

    // Load weights into registers (done once, small)
    float w[d_conv] = { 0.0f };
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    // Compute from shared memory
    for (int64_t i = 0; i < local_n_t; i++) {
        float sumf = 0.0f;
#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += smem[tid * n_cols + i + j] * w[j];
        }
        sumf += b;
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

// Fused decode-time depthwise causal conv1d update (one new token). Each thread owns one channel of
// one sequence: it assembles the width-d_conv window from the K-1 cached taps (conv_states) plus the
// current token (x_cur), computes the depthwise conv with the SAME ascending-tap FMA order as
// ssm_conv_f32 at i==0, optionally folds silu, writes the conv output, and writes the 1-token-shifted
// ring state back in place into conv_state_dst. Bit-identical to ssm_conv(concat) + silu + copy-back.
template <bool apply_silu, int d_conv>
static __global__ void ssm_conv_update_f32(const float * __restrict__ conv_states,
                                           const float * __restrict__ conv_kernel,
                                           const float * __restrict__ x_cur,
                                           float       * __restrict__ conv_state_dst,
                                           float       * __restrict__ dst,
                                           const int channels,
                                           const int states_seq_stride,
                                           const int w_stride,
                                           const int x_seq_stride,
                                           const int dst_seq_stride,
                                           const int cdst_seq_stride) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x; // channel
    const int s = blockIdx.y;                            // sequence
    if (c >= channels) {
        return;
    }

    const float * states_c = conv_states + (int64_t) s * states_seq_stride + (int64_t) c * (d_conv - 1);
    const float * w_c       = conv_kernel + (int64_t) c * w_stride;
    const float   xc        = x_cur[(int64_t) s * x_seq_stride + c];

    // window = [tap0 .. tap_{K-2}, current-token], same ordering as the concat(conv_states, x) window
    float window[d_conv];
#pragma unroll
    for (int j = 0; j < d_conv - 1; j++) {
        window[j] = states_c[j];
    }
    window[d_conv - 1] = xc;

    float sumf = 0.0f;
#pragma unroll
    for (int j = 0; j < d_conv; j++) {
        sumf += window[j] * w_c[j];
    }
    sumf += 0.0f; // matches ssm_conv_f32 `sumf += b` with b == 0 (qwen35 conv1d has no bias)
    dst[(int64_t) s * dst_seq_stride + c] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;

    // 1-token-shifted ring write-back: drop the oldest tap, append the current token
    float * out_state = conv_state_dst + (int64_t) s * cdst_seq_stride + (int64_t) c * (d_conv - 1);
#pragma unroll
    for (int j = 0; j < d_conv - 1; j++) {
        out_state[j] = window[j + 1];
    }
}

static void ggml_cuda_op_ssm_conv_update(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * conv_states = dst->src[0]; // [K-1, channels, n_seqs]
    const ggml_tensor * conv_kernel = dst->src[1]; // [K, channels]
    const ggml_tensor * x_cur       = dst->src[2]; // [channels, 1, n_seqs]
    const ggml_tensor * cdst        = dst->src[3]; // [(K-1)*channels, n_seqs] in-place ring target

    const int64_t d_conv   = conv_kernel->ne[0];
    const int64_t channels = conv_kernel->ne[1];
    const int64_t n_seqs   = conv_states->ne[2];
    const bool    apply_silu = ggml_get_op_params_i32(dst, 0) != 0;

    GGML_ASSERT(conv_states->type == GGML_TYPE_F32 && conv_kernel->type == GGML_TYPE_F32);
    GGML_ASSERT(x_cur->type == GGML_TYPE_F32 && cdst->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(conv_states->nb[0] == sizeof(float));
    GGML_ASSERT(conv_states->nb[1] == (size_t) (d_conv - 1) * sizeof(float));
    GGML_ASSERT(conv_kernel->nb[0] == sizeof(float));
    GGML_ASSERT(dst->ne[0] == channels && dst->ne[1] == 1 && dst->ne[2] == n_seqs);

    const float * states_d = (const float *) conv_states->data;
    const float * w_d      = (const float *) conv_kernel->data;
    const float * x_d      = (const float *) x_cur->data;
    float *       cdst_d   = (float *) cdst->data;
    float *       dst_d    = (float *) dst->data;
    cudaStream_t  stream   = ctx.stream();

    const int states_seq_stride = (int) (conv_states->nb[2] / sizeof(float));
    const int w_stride          = (int) (conv_kernel->nb[1] / sizeof(float));
    const int x_seq_stride      = (int) (x_cur->nb[2] / sizeof(float));
    const int dst_seq_stride    = (int) (dst->nb[2] / sizeof(float));
    const int cdst_seq_stride   = (int) (cdst->nb[1] / sizeof(float));

    const int threads = 128;
    const dim3 blocks((channels + threads - 1) / threads, (unsigned) n_seqs, 1);

    auto launch = [&](auto NC) {
        constexpr int kNC = decltype(NC)::value;
        if (apply_silu) {
            ssm_conv_update_f32<true, kNC><<<blocks, threads, 0, stream>>>(states_d, w_d, x_d, cdst_d, dst_d,
                (int) channels, states_seq_stride, w_stride, x_seq_stride, dst_seq_stride, cdst_seq_stride);
        } else {
            ssm_conv_update_f32<false, kNC><<<blocks, threads, 0, stream>>>(states_d, w_d, x_d, cdst_d, dst_d,
                (int) channels, states_seq_stride, w_stride, x_seq_stride, dst_seq_stride, cdst_seq_stride);
        }
    };

    switch (d_conv) {
        case 3: launch(std::integral_constant<int, 3>{}); break;
        case 4: launch(std::integral_constant<int, 4>{}); break;
        default: GGML_ABORT("ssm_conv_update only supports d_conv 3 or 4");
    }
}

// Patch 0028: gather only the NON-identity sequences' prior conv taps from the FULL conv cache into a
// disjoint scratch buffer. Identity sequences (ids[s] == rs_head + s) are read in place from the
// destination slot by the update kernel and are skipped here. One block per sequence. Mirrors
// gdn_gather_nonident_kernel (the 0019 recurrent-state gather fusion).
static __global__ void ssm_conv_gather_nonident_kernel(const float * __restrict__ cache,
                                                       const int32_t * __restrict__ ids, int rs_head,
                                                       float * __restrict__ scratch, int row_stride, int n_seqs) {
    const int s = blockIdx.x;
    if (s >= n_seqs) {
        return;
    }
    const int r = ids[s];
    if (r == rs_head + s) {
        return; // identity: prior taps already live in the in-place destination slot
    }
    const float * src = cache   + (int64_t) r * row_stride;
    float *       dst = scratch + (int64_t) s * row_stride;
    for (int i = threadIdx.x; i < row_stride; i += blockDim.x) {
        dst[i] = src[i];
    }
}

// Patch 0028: gather-free fused conv update. Per (channel, sequence), read the K-1 prior taps from the
// active sequence's cache slot via ids -- identity (ids[s] == rs_head + s) reads in place from
// conv_state_dst (the same slot it writes; the whole window is loaded into registers before any write,
// so it is race-free), non-identity reads the pre-gathered disjoint scratch -- then computes the
// depthwise conv with the SAME ascending-tap FMA order as ssm_conv_update_f32, folds silu, writes the
// conv output, and writes the 1-token-shifted ring state back in place. Bit-identical to the get_rows +
// ssm_conv_update_f32 path: the read VALUES are the same; only the read POINTER changes.
template <bool apply_silu, int d_conv>
static __global__ void ssm_conv_update_ids_f32(const float * __restrict__ nonident_scratch,
                                               const float * __restrict__ conv_kernel,
                                               const float * __restrict__ x_cur,
                                               float       * __restrict__ conv_state_dst,
                                               float       * __restrict__ dst,
                                               const int32_t * __restrict__ ids,
                                               const int   rs_head,
                                               const int   channels,
                                               const int   scratch_seq_stride,
                                               const int   w_stride,
                                               const int   x_seq_stride,
                                               const int   dst_seq_stride,
                                               const int   cdst_seq_stride) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x; // channel
    const int s = blockIdx.y;                            // sequence
    if (c >= channels) {
        return;
    }

    const bool ident = (ids[s] == rs_head + s);
    const float * states_c = ident
        ? conv_state_dst   + (int64_t) s * cdst_seq_stride    + (int64_t) c * (d_conv - 1)
        : nonident_scratch + (int64_t) s * scratch_seq_stride + (int64_t) c * (d_conv - 1);
    const float * w_c = conv_kernel + (int64_t) c * w_stride;
    const float   xc  = x_cur[(int64_t) s * x_seq_stride + c];

    // window = [tap0 .. tap_{K-2}, current-token], same ordering as ssm_conv_update_f32
    float window[d_conv];
#pragma unroll
    for (int j = 0; j < d_conv - 1; j++) {
        window[j] = states_c[j];
    }
    window[d_conv - 1] = xc;

    float sumf = 0.0f;
#pragma unroll
    for (int j = 0; j < d_conv; j++) {
        sumf += window[j] * w_c[j];
    }
    sumf += 0.0f; // matches ssm_conv_f32 `sumf += b` with b == 0 (qwen35 conv1d has no bias)
    dst[(int64_t) s * dst_seq_stride + c] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;

    // 1-token-shifted ring write-back: drop the oldest tap, append the current token
    float * out_state = conv_state_dst + (int64_t) s * cdst_seq_stride + (int64_t) c * (d_conv - 1);
#pragma unroll
    for (int j = 0; j < d_conv - 1; j++) {
        out_state[j] = window[j + 1];
    }
}

static void ggml_cuda_op_ssm_conv_update_ids(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * conv_states = dst->src[0]; // FULL cache [K-1, channels, n_cells]
    const ggml_tensor * conv_kernel = dst->src[1]; // [K, channels]
    const ggml_tensor * x_cur       = dst->src[2]; // [channels, 1, n_seqs]
    const ggml_tensor * cdst        = dst->src[3]; // [(K-1)*channels, n_seqs] in-place ring target
    const ggml_tensor * ids         = dst->src[4]; // [n_seqs] I32 slot indices (s_copy)

    const int64_t d_conv   = conv_kernel->ne[0];
    const int64_t channels = conv_kernel->ne[1];
    const int64_t n_seqs   = x_cur->ne[2];
    const bool    apply_silu = ggml_get_op_params_i32(dst, 0) != 0;
    const int     rs_head    = ggml_get_op_params_i32(dst, 1);

    GGML_ASSERT(conv_states->type == GGML_TYPE_F32 && conv_kernel->type == GGML_TYPE_F32);
    GGML_ASSERT(x_cur->type == GGML_TYPE_F32 && cdst->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ids->type == GGML_TYPE_I32);
    GGML_ASSERT(conv_states->nb[0] == sizeof(float));
    GGML_ASSERT(conv_states->nb[1] == (size_t) (d_conv - 1) * sizeof(float));
    GGML_ASSERT(conv_kernel->nb[0] == sizeof(float));
    GGML_ASSERT(dst->ne[0] == channels && dst->ne[1] == 1 && dst->ne[2] == n_seqs);

    const float *   cache_d = (const float *) conv_states->data;
    const float *   w_d     = (const float *) conv_kernel->data;
    const float *   x_d     = (const float *) x_cur->data;
    float *         cdst_d  = (float *) cdst->data;
    float *         dst_d   = (float *) dst->data;
    const int32_t * ids_d   = (const int32_t *) ids->data;
    cudaStream_t    stream  = ctx.stream();

    // n_embd_r = (K-1)*channels: the per-cell row stride of the full conv cache.
    const int cache_row_stride = (int) (conv_states->nb[2] / sizeof(float));
    const int w_stride         = (int) (conv_kernel->nb[1] / sizeof(float));
    const int x_seq_stride     = (int) (x_cur->nb[2] / sizeof(float));
    const int dst_seq_stride   = (int) (dst->nb[2] / sizeof(float));
    const int cdst_seq_stride  = (int) (cdst->nb[1] / sizeof(float));

    // Gather only the non-identity sequences' prior taps into a disjoint scratch (identity sequences
    // read in place from cdst). The scratch is written here and read-only by the update kernel, so the
    // update kernel never reads a slot another block writes -> race-free. No-op at steady AR decode.
    ggml_cuda_pool_alloc<float> nonident_scratch(ctx.pool());
    float * scratch = nonident_scratch.alloc((size_t) cache_row_stride * n_seqs);
    if (n_seqs > 0) {
        ssm_conv_gather_nonident_kernel<<<(unsigned) n_seqs, 256, 0, stream>>>(
            cache_d, ids_d, rs_head, scratch, cache_row_stride, (int) n_seqs);
    }

    const int threads = 128;
    const dim3 blocks((channels + threads - 1) / threads, (unsigned) n_seqs, 1);

    auto launch = [&](auto NC) {
        constexpr int kNC = decltype(NC)::value;
        if (apply_silu) {
            ssm_conv_update_ids_f32<true, kNC><<<blocks, threads, 0, stream>>>(scratch, w_d, x_d, cdst_d, dst_d,
                ids_d, rs_head, (int) channels, cache_row_stride, w_stride, x_seq_stride, dst_seq_stride, cdst_seq_stride);
        } else {
            ssm_conv_update_ids_f32<false, kNC><<<blocks, threads, 0, stream>>>(scratch, w_d, x_d, cdst_d, dst_d,
                ids_d, rs_head, (int) channels, cache_row_stride, w_stride, x_seq_stride, dst_seq_stride, cdst_seq_stride);
        }
    };

    switch (d_conv) {
        case 3: launch(std::integral_constant<int, 3>{}); break;
        case 4: launch(std::integral_constant<int, 4>{}); break;
        default: GGML_ABORT("ssm_conv_update_ids only supports d_conv 3 or 4");
    }
}

template <bool apply_silu>
static void ssm_conv_f32_cuda(const float * src0, const float * src1, const float * bias, const int src0_nb0, const int src0_nb1,
                              const int src0_nb2, const int src1_nb1, float * dst, const int dst_nb0, const int dst_nb1,
                              const int dst_nb2, const int64_t nc, const int64_t nr, const int64_t n_t,
                              const int64_t n_s, cudaStream_t stream) {
    const int threads = 128;
    GGML_ASSERT(nr % threads == 0);

    auto launch_kernel = [&](auto NC) {
        constexpr int kNC = decltype(NC)::value;
        if (n_t <= 32) {
            const dim3 blocks(n_s, (nr + threads - 1) / threads, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks, threads, 0, stream);
            ggml_cuda_kernel_launch(ssm_conv_f32<apply_silu, threads, kNC>, launch_params, src0, src1, bias, src0_nb0, src0_nb1,
                                                                        src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        } else {
            const int64_t split_n_t = 32;
            dim3          blocks(n_s, (nr + threads - 1) / threads, (n_t + split_n_t - 1) / split_n_t);
            const size_t  smem_size = threads * (kNC - 1 + split_n_t) * sizeof(float);
            ssm_conv_long_token_f32<apply_silu, threads, kNC, split_n_t><<<blocks, threads, smem_size, stream>>>(
                src0, src1, bias, src0_nb0, src0_nb1, src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        }
    };

    switch (nc) {
        case 3:  launch_kernel(std::integral_constant<int, 3 >{}); break;
        case 4:  launch_kernel(std::integral_constant<int, 4 >{}); break;
        case 5:  launch_kernel(std::integral_constant<int, 5 >{}); break;
        case 9:  launch_kernel(std::integral_constant<int, 9 >{}); break;
        case 15: launch_kernel(std::integral_constant<int, 15>{}); break;
        default: GGML_ABORT("Only support kernel sizes 3, 4, 5, 9, 15 right now.");
    }
}

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node, ggml_tensor * silu_dst) {
    // Fused decode conv-update-in-place variant (ggml_ssm_conv_update_inplace): discriminated by a
    // non-null src[3] (the in-place ring write-back target). It folds the concat/transpose/copy-back/
    // silu of the decode conv path into a single kernel.
    if (dst->src[3] != nullptr) {
        GGML_ASSERT(bias_add_node == nullptr && silu_dst == nullptr);
        // Patch 0028: a non-null src[4] (ids) selects the gather-free variant that reads each
        // sequence's prior taps directly from the full cache via ids (no get_rows materialization).
        if (dst->src[4] != nullptr) {
            ggml_cuda_op_ssm_conv_update_ids(ctx, dst);
        } else {
            ggml_cuda_op_ssm_conv_update(ctx, dst);
        }
        return;
    }

    const struct ggml_tensor * src0 = dst->src[0];  // conv_x
    const struct ggml_tensor * src1 = dst->src[1];  // conv1d.weight
    const bool fuse_bias = bias_add_node != nullptr;
    const bool fuse_silu = silu_dst != nullptr;

    // bias always comes with silu.
    GGML_ASSERT(!fuse_bias || fuse_silu);

    // The bias (when fused) is the non-conv operand of the ADD node.
    const struct ggml_tensor * bias = fuse_bias ? (bias_add_node->src[0] == dst ? bias_add_node->src[1] : bias_add_node->src[0]) : nullptr;

    // When fusing, write to silu_dst (the node downstream references).
    const struct ggml_tensor * out = fuse_silu ? silu_dst : dst;

    const int64_t nc  = src1->ne[0];                // d_conv
    const int64_t nr  = src0->ne[1];                // d_inner
    const int64_t n_t = out->ne[1];                 // tokens per sequence
    const int64_t n_s = out->ne[2];                 // number of sequences in the batch

    GGML_ASSERT(out->ne[0] == nr);
    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(src0->nb[1] == src0->ne[0] * sizeof(float));

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    const float * bias_d = fuse_bias ? (const float *) bias->data : nullptr;
    float *       dst_d  = (float *) out->data;
    cudaStream_t  stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(out->type == GGML_TYPE_F32);
    if (fuse_bias) {
        GGML_ASSERT(bias->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(bias));
        GGML_ASSERT(ggml_nelements(bias) == nr);
    }

    if (fuse_silu) {
        ssm_conv_f32_cuda<true>(src0_d, src1_d, bias_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream);
    } else {
        ssm_conv_f32_cuda<false>(src0_d, src1_d, bias_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream);
    }
}
