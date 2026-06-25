#include "gated_delta_net.cuh"
#include "ggml-cuda/common.cuh"

#include <cstdlib>

// Step 2: gather only the NON-identity sequences' prior recurrent state from the full cache into a
// disjoint scratch buffer. Identity sequences (ids[s] == rs_head + s) are read in place from the
// destination slot by the recurrence kernel and are skipped here. One block per sequence.
__global__ void gdn_gather_nonident_kernel(const float * cache, const int32_t * ids, int rs_head,
                                           float * scratch, int64_t D, int n_seqs) {
    const int s = blockIdx.x;
    if (s >= n_seqs) {
        return;
    }
    const int r = ids[s];
    if (r == rs_head + s) {
        return; // identity: prior state already lives in the in-place destination slot
    }
    const float * src = cache   + (int64_t) r * D;
    float *       dst = scratch + (int64_t) s * D;
    for (int64_t i = threadIdx.x; i < D; i += blockDim.x) {
        dst[i] = src[i];
    }
}

static void ggml_cuda_gdn_gather_nonident(const float * cache, const int32_t * ids, int rs_head,
                                          float * scratch, int64_t D, int64_t n_seqs, cudaStream_t stream) {
    if (n_seqs <= 0) {
        return;
    }
    gdn_gather_nonident_kernel<<<(unsigned) n_seqs, 256, 0, stream>>>(cache, ids, rs_head, scratch, D, (int) n_seqs);
}

// Occupancy/coalescing retune (patch 0022). Each warp owns COLS_PER_WARP columns of the recurrent
// state instead of 1, looping the existing per-column body over col, col+NUM_WARPS, ... within a
// per-block column tile of size NUM_WARPS*COLS_PER_WARP. The S_v rows of every column stay sharded
// across the lanes by the SAME strided mapping i = r*warp_size + lane, and every column's per-lane
// FMA accumulation and warp_reduce_sum<warp_size> butterfly are byte-for-byte unchanged. Only the
// (warp,block)->column assignment and the order a warp visits its columns differ, and a column's
// f32 value provably does not depend on either (columns are fully independent: column c reads only
// its own S_v-float state slice plus the shared per-(token,head,seq) q/k/v/g/beta). So the result
// and the stored final state are bit-identical to the COLS_PER_WARP==1 baseline (md5-gateable),
// while per-warp memory-level parallelism rises ~COLS_PER_WARP-fold (COLS_PER_WARP independent
// state-load bursts issued before any reduction, and the independent butterfly reductions interleave
// to hide each other's shfl latency) which covers more DRAM latency on this bandwidth-bound kernel.
// Every individual global access stays IDENTICALLY coalesced (32 consecutive lanes -> one 128B
// sector), so this is a latency-coverage / scheduling win, not a coalescing change.
template <int S_v, bool KDA, bool keep_rs_t, int NUM_WARPS = 4, int COLS_PER_WARP = 1, int MIN_BLOCKS = 2>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * NUM_WARPS, MIN_BLOCKS)
gated_delta_net_cuda(const float * q,
                                     const float * k,
                                     const float * v,
                                     const float * g,
                                     const float * beta,
                                     const float * curr_state,
                                     float *       dst,
                                     int64_t       H,
                                     int64_t       n_tokens,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale,
                                     int           K,
                                     float *       state_dst,
                                     const int32_t * ids,
                                     int           rs_head) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    // each warp owns COLS_PER_WARP columns, using warp-level primitives to reduce across rows.
    const int      lane     = threadIdx.x;
    const int      col_base = blockIdx.z * (NUM_WARPS * COLS_PER_WARP) + threadIdx.y;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    const int64_t attn_score_elems = S_v * H * n_tokens * n_seqs;
    float *       attn_data        = dst;
    // when state_dst is provided (in-place decode write-back) the final recurrent state is written
    // directly into the persistent cache view instead of being appended to the op output; this
    // eliminates the per-layer per-step D2D state copy-back. Only used when keep_rs_t == false.
    float *       state            = (state_dst != nullptr) ? state_dst : (dst + attn_score_elems);

    // input state holds s0 only: [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    const int64_t state_in_offset      = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    // Step 2: select the prior-state read base per sequence. For the ids variant, identity
    // sequences (ids[seq] == rs_head + seq) read s0 directly from the in-place destination slot
    // state_dst (no materialization); non-identity sequences read from the pre-gathered scratch
    // (curr_state). state_in_offset == state_out_offset, so both bases use the same per-(seq,head)
    // offset. The whole s0 is loaded into registers before the new state is written, so reading and
    // writing the same slot per block (identity) is race-free.
    const float * read_state = (ids != nullptr && ids[sequence] == rs_head + (int) sequence)
        ? state_dst : curr_state;
    read_state += state_in_offset;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
    // per-column register shard of the recurrent state; state is stored transposed: M[col][i] = S[i][col].
    float         s_shard[COLS_PER_WARP][rows_per_lane];

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int cc = 0; cc < COLS_PER_WARP; cc++) {
        const int     col = col_base + cc * NUM_WARPS;
        const float * rs  = read_state + col * S_v;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i   = r * warp_size + lane;
            s_shard[cc][r] = rs[i];
        }
    }

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        const float beta_val = *beta_t;

        // Cache k and q in registers (shared across the COLS_PER_WARP columns of this warp).
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
        }

        if constexpr (!KDA) {
            const float g_val = expf(*g_t);

#pragma unroll
            for (int cc = 0; cc < COLS_PER_WARP; cc++) {
                const int col = col_base + cc * NUM_WARPS;

                // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
                float kv_shard = 0.0f;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    kv_shard += s_shard[cc][r] * k_reg[r];
                }
                float kv_col = warp_reduce_sum<warp_size>(kv_shard);

                // delta[col] = (v[col] - g * kv[col]) * beta
                float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

                // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
                // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
                float attn_partial = 0.0f;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    s_shard[cc][r]  = g_val * s_shard[cc][r] + k_reg[r] * delta_col;
                    attn_partial += s_shard[cc][r] * q_reg[r];
                }

                float attn_col = warp_reduce_sum<warp_size>(attn_partial);

                if (lane == 0) {
                    attn_data[col] = attn_col * scale;
                }
            }
        } else {
#pragma unroll
            for (int cc = 0; cc < COLS_PER_WARP; cc++) {
                const int col = col_base + cc * NUM_WARPS;

                // kv[col] = sum_i g[i] * S[i][col] * k[i]
                float kv_shard = 0.0f;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * warp_size + lane;
                    kv_shard += expf(g_t[i]) * s_shard[cc][r] * k_reg[r];
                }

                float kv_col = warp_reduce_sum<warp_size>(kv_shard);

                // delta[col] = (v[col] - kv[col]) * beta
                float delta_col = (v_t[col] - kv_col) * beta_val;

                // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
                // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
                float attn_partial = 0.0f;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * warp_size + lane;
                    s_shard[cc][r]  = expf(g_t[i]) * s_shard[cc][r] + k_reg[r] * delta_col;
                    attn_partial += s_shard[cc][r] * q_reg[r];
                }

                float attn_col = warp_reduce_sum<warp_size>(attn_partial);

                if (lane == 0) {
                    attn_data[col] = attn_col * scale;
                }
            }
        }

        attn_data += S_v * H;

        if constexpr (keep_rs_t) {
            // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
            // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.
            const int64_t state_size_per_token = S_v * S_v * H * n_seqs; // per-slot stride in output
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
#pragma unroll
                for (int cc = 0; cc < COLS_PER_WARP; cc++) {
                    const int col = col_base + cc * NUM_WARPS;
                    float * curr_state = (dst + attn_score_elems) + target_slot * state_size_per_token + state_out_offset;
#pragma unroll
                    for (int r = 0; r < rows_per_lane; r++) {
                        const int i = r * warp_size + lane;
                        curr_state[col * S_v + i] = s_shard[cc][r];
                    }
                }
            }
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int cc = 0; cc < COLS_PER_WARP; cc++) {
            const int col = col_base + cc * NUM_WARPS;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i          = r * warp_size + lane;
                state[col * S_v + i] = s_shard[cc][r];
            }
        }
    }
}

// Default column-folding tile for the S_v==128 decode/prefill path (the GDN head dim of this model).
// Measured winner of the bit-exact occupancy sweep (patch 0022). Override at runtime for the sweep
// via GDN_NW / GDN_CPW; all selectable variants are bit-identical, only %peak differs.
#ifndef GDN_DEFAULT_NW
#define GDN_DEFAULT_NW 16
#endif
#ifndef GDN_DEFAULT_CPW
#define GDN_DEFAULT_CPW 8
#endif

template <int S_v, bool KDA, bool keep_rs_t, int NUM_WARPS, int COLS_PER_WARP, int MIN_BLOCKS>
static void launch_gdn_variant(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_dst_d, const int32_t * ids_d, int rs_head,
        int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1, int64_t sq2, int64_t sq3,
        int64_t sv1, int64_t sv2, int64_t sv3,
        int64_t sb1, int64_t sb2, int64_t sb3,
        const uint3 neqk1_magic, const uint3 rq3_magic,
        float scale, int K, int warp_size, cudaStream_t stream) {
    static_assert(S_v % (NUM_WARPS * COLS_PER_WARP) == 0, "NUM_WARPS*COLS_PER_WARP must divide S_v");
    dim3 grid_dims(H, n_seqs, S_v / (NUM_WARPS * COLS_PER_WARP));
    dim3 block_dims(warp_size <= S_v ? warp_size : S_v, NUM_WARPS, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    ggml_cuda_kernel_launch(gated_delta_net_cuda<S_v, KDA, keep_rs_t, NUM_WARPS, COLS_PER_WARP, MIN_BLOCKS>, launch_params,
        q_d, k_d, v_d, g_d, b_d, s_d, dst_d, H,
        n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
        sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, K, state_dst_d, ids_d, rs_head);
}

template <bool KDA, bool keep_rs_t>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_dst_d,
        const int32_t * ids_d, int rs_head,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int K, cudaStream_t stream) {
    //TODO: Add chunked kernel for even faster pre-fill
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

#define GDN_LAUNCH_ARGS \
        q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_dst_d, ids_d, rs_head, \
        H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, \
        neqk1_magic, rq3_magic, scale, K, warp_size, stream

    switch (S_v) {
        case 16:
            launch_gdn_variant<16, KDA, keep_rs_t, 4, 1, 2>(GDN_LAUNCH_ARGS);
            break;
        case 32:
            launch_gdn_variant<32, KDA, keep_rs_t, 4, 1, 2>(GDN_LAUNCH_ARGS);
            break;
        case 64:
            launch_gdn_variant<64, KDA, keep_rs_t, 4, 1, 2>(GDN_LAUNCH_ARGS);
            break;
        case 128: {
            // Bit-exact occupancy/coalescing retune (patch 0022): fold COLS_PER_WARP columns per warp
            // to raise per-warp memory-level parallelism on this bandwidth-bound recurrence. Default is
            // the measured winner; GDN_NW / GDN_CPW override it for the one-build %peak sweep (every
            // selectable {num_warps, cols} is bit-identical, so the sweep cannot change the md5).
            static const int gdn_nw  = []{ const char * e = getenv("GDN_NW");  return e ? atoi(e) : GDN_DEFAULT_NW;  }();
            static const int gdn_cpw = []{ const char * e = getenv("GDN_CPW"); return e ? atoi(e) : GDN_DEFAULT_CPW; }();
            // NUM_WARPS in {4,8,16} x COLS_PER_WARP ladder (all <=512 threads/block, no 1024-thread
            // .minnctapersm warnings). Measured GB10 %peak: (4,1)=73 baseline ... (16,4)=82 ...
            // (16,8)=84.7 winner ~ tied with (8,8)/(8,16)/(32,4); the plateau is just above vLLM (82.4).
            if      (gdn_nw == 4  && gdn_cpw == 1) launch_gdn_variant<128, KDA, keep_rs_t, 4,  1, 2>(GDN_LAUNCH_ARGS);
            else if (gdn_nw == 4  && gdn_cpw == 2) launch_gdn_variant<128, KDA, keep_rs_t, 4,  2, 2>(GDN_LAUNCH_ARGS);
            else if (gdn_nw == 4  && gdn_cpw == 4) launch_gdn_variant<128, KDA, keep_rs_t, 4,  4, 2>(GDN_LAUNCH_ARGS);
            else if (gdn_nw == 8  && gdn_cpw == 1) launch_gdn_variant<128, KDA, keep_rs_t, 8,  1, 2>(GDN_LAUNCH_ARGS);
            else if (gdn_nw == 8  && gdn_cpw == 2) launch_gdn_variant<128, KDA, keep_rs_t, 8,  2, 2>(GDN_LAUNCH_ARGS);
            else if (gdn_nw == 8  && gdn_cpw == 4) launch_gdn_variant<128, KDA, keep_rs_t, 8,  4, 2>(GDN_LAUNCH_ARGS);
            else if (gdn_nw == 8  && gdn_cpw == 8) launch_gdn_variant<128, KDA, keep_rs_t, 8,  8, 2>(GDN_LAUNCH_ARGS);
            else if (gdn_nw == 16 && gdn_cpw == 1) launch_gdn_variant<128, KDA, keep_rs_t, 16, 1, 2>(GDN_LAUNCH_ARGS);
            else if (gdn_nw == 16 && gdn_cpw == 2) launch_gdn_variant<128, KDA, keep_rs_t, 16, 2, 2>(GDN_LAUNCH_ARGS);
            else if (gdn_nw == 16 && gdn_cpw == 4) launch_gdn_variant<128, KDA, keep_rs_t, 16, 4, 2>(GDN_LAUNCH_ARGS);
            else if (gdn_nw == 16 && gdn_cpw == 8) launch_gdn_variant<128, KDA, keep_rs_t, 16, 8, 2>(GDN_LAUNCH_ARGS);
            else                                   launch_gdn_variant<128, KDA, keep_rs_t, GDN_DEFAULT_NW, GDN_DEFAULT_CPW, 2>(GDN_LAUNCH_ARGS);
            break;
        }
        default:
            GGML_ABORT("fatal error");
            break;
    }

#undef GDN_LAUNCH_ARGS
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];
    ggml_tensor * src_state_dst = dst->src[6]; // optional in-place state write-back target

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    float *       dst_d = (float *) dst->data;

    float * state_dst_d = nullptr;
    if (src_state_dst != nullptr) {
        // in-place final-state cache view: per-seq stride must be the dense state size D = S_v*S_v*H
        GGML_ASSERT(src_state_dst->type == GGML_TYPE_F32);
        GGML_ASSERT(src_state_dst->nb[0] == sizeof(float));
        GGML_ASSERT(src_state_dst->nb[1] == (size_t) S_v * S_v * H * sizeof(float));
        state_dst_d = (float *) src_state_dst->data;
    }

    // Step 2: fused recurrent-state gather (src[7] = ids == s_copy). Read the prior state directly
    // from the full cache via ids instead of from a materialized ggml_get_rows gather. The recurrence
    // kernel reads identity sequences (ids[seq] == rs_head + seq) in place from state_dst (no
    // materialization at all); any non-identity sequence (reorder / rs_zero remap) is gathered here
    // into a disjoint scratch that the kernel reads instead. The gather writes a disjoint buffer and
    // the recurrence never reads a slot another block writes, so it is race-free and bit-identical to
    // the get_rows path. ids stays a DEVICE pointer (dereferenced only inside the kernels).
    ggml_tensor * src_ids = dst->src[7];
    const float *   s_d     = (const float *) src_state->data;
    const int32_t * ids_d   = nullptr;
    int             rs_head = 0;
    ggml_cuda_pool_alloc<float> ids_state_scratch(ctx.pool());
    if (src_ids != nullptr) {
        GGML_ASSERT(state_dst_d != nullptr);
        GGML_ASSERT(src_ids->type == GGML_TYPE_I32);
        rs_head = ggml_get_op_params_i32(dst, 1);
        ids_d   = (const int32_t *) src_ids->data;
        const int64_t D = S_v * S_v * H;
        float * scratch = ids_state_scratch.alloc((size_t) D * n_seqs);
        ggml_cuda_gdn_gather_nonident(s_d, ids_d, rs_head, scratch, D, n_seqs, ctx.stream());
        s_d = scratch;
    }

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    const int K = ggml_get_op_params_i32(dst, 0);
    const bool keep_rs = K > 1;

    // in-place write-back is only valid for the single-snapshot (final-state) case
    GGML_ASSERT(state_dst_d == nullptr || !keep_rs);

    if (kda) {
        if (keep_rs) {
            launch_gated_delta_net<true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_dst_d, ids_d, rs_head,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, K, stream);
        } else {
            launch_gated_delta_net<true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_dst_d, ids_d, rs_head,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, K, stream);
        }
    } else {
        if (keep_rs) {
            launch_gated_delta_net<false, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_dst_d, ids_d, rs_head,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, K, stream);
        } else {
            launch_gated_delta_net<false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_dst_d, ids_d, rs_head,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, K, stream);
        }
    }
}
