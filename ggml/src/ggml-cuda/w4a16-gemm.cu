#include "w4a16-gemm.cuh"
#include "mma.cuh"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

// ===========================================================================
// [paged patch 0035] Marlin-style W4A16 grouped MoE prefill GEMM. See w4a16-gemm.cuh.
//
// In-register FP4->bf16 weight dequant + bf16 activations + bf16 m16n8k16 mma.sync (mma.cuh),
// cp.async multistage, grouped (ragged, per-tile expert offset) over the token-sorted buffer.
// ===========================================================================

using namespace ggml_cuda_mma;
typedef tile<16, 8, nv_bfloat162> tile_A; // A operand: M=16, K=16
typedef tile< 8, 8, nv_bfloat162> tile_B; // B operand: N=8,  K=16
typedef tile<16, 8, float>        tile_C; // accumulator: M=16, N=8

struct alignas(16) w4a16_tile_desc {
    int expert;
    int row0;
    int rows;
    int pad;
};
static_assert(sizeof(w4a16_tile_desc) == 16, "w4a16 tile descriptors must stay 16 bytes");

#ifndef LLAMA_W4A16_PREFILL_M
#define LLAMA_W4A16_PREFILL_M 0
#endif // LLAMA_W4A16_PREFILL_M

int64_t ggml_cuda_w4a16_prefill_m() {
    static const int64_t m = [] {
        const char * e = getenv("LLAMA_W4A16_PREFILL_M");
        return e != nullptr ? (int64_t) atoll(e) : (int64_t) LLAMA_W4A16_PREFILL_M;
    }();
    return m;
}

bool ggml_cuda_w4a16_prefill_enabled() {
    return ggml_cuda_w4a16_prefill_m() > 0;
}

// ---- cp.async helpers (sm80+; raw bytes, no cast) ----
static __device__ __forceinline__ void w4a16_cp_async16(void * smem, const void * gmem) {
#ifdef CP_ASYNC_AVAILABLE
    const unsigned s = (unsigned) __cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0],[%1],16;\n" :: "r"(s), "l"(gmem));
#else
    GGML_UNUSED(smem); GGML_UNUSED(gmem); NO_DEVICE_CODE;
#endif // CP_ASYNC_AVAILABLE
}
static __device__ __forceinline__ void w4a16_cp_async4(void * smem, const void * gmem) {
#ifdef CP_ASYNC_AVAILABLE
    const unsigned s = (unsigned) __cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0],[%1],4;\n" :: "r"(s), "l"(gmem));
#else
    GGML_UNUSED(smem); GGML_UNUSED(gmem); NO_DEVICE_CODE;
#endif // CP_ASYNC_AVAILABLE
}
static __device__ __forceinline__ void w4a16_cp_commit() {
#ifdef CP_ASYNC_AVAILABLE
    asm volatile("cp.async.commit_group;\n" ::);
#else
    NO_DEVICE_CODE;
#endif // CP_ASYNC_AVAILABLE
}
template<int N> static __device__ __forceinline__ void w4a16_cp_wait() {
#ifdef CP_ASYNC_AVAILABLE
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
#else
    NO_DEVICE_CODE;
#endif // CP_ASYNC_AVAILABLE
}

// ---- f32 -> bf16 activation cast (NO quantize). Pads the [total_rows, pad_rows) tail with 0. ----
static __global__ void w4a16_cast_act_f32_bf16(
        const float * __restrict__ x, nv_bfloat16 * __restrict__ y, int64_t n, int64_t npad) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= npad) {
        return;
    }
    y[i] = i < n ? __float2bfloat16(x[i]) : (nv_bfloat16) 0.0f;
}

// ---------------------------------------------------------------------------
// Grouped W4A16 GEMM. For each output tile (blockIdx.x = N-block, blockIdx.y = M-tile):
//   expert e   = g_tile_expert[blockIdx.y]
//   row_start  = g_tile_row0[blockIdx.y]   (absolute row in the sorted buffer)
//   row_count  = g_tile_rows[blockIdx.y]   (valid rows in this tile, <= BM)
// Weights read from W = src0 + e*expert_stride_blocks (block_nvfp4 [N,Kb]); activations from
// Abf (bf16, sorted); output to C (f32, sorted, [N, total_rows] = C[row*N + col]).
// Weights are dequantized FP4->bf16 in registers; A via ldmatrix; bf16 m16n8k16 mma.
// BK = 64 (one nvfp4 block per K-step); STAGES-deep cp.async pipeline over the Kb blocks.
// ---------------------------------------------------------------------------
template<int BM, int BN, int WARPS_M, int WARPS_N, int STAGES>
__launch_bounds__(WARPS_M*WARPS_N*32, 1)
static __global__ void w4a16_grouped_kernel(
        const nv_bfloat16 * __restrict__ Abf,            // [pad_rows, K] bf16
        const block_nvfp4 * __restrict__ W0,             // src0 base (expert 0)
        float * __restrict__ C,                          // [total_rows, N] f32
        const w4a16_tile_desc * __restrict__ g_tiles,
        int N, int K, int64_t expert_stride_blocks) {
#if defined(AMPERE_MMA_AVAILABLE) && defined(CP_ASYNC_AVAILABLE)
    constexpr int BK   = 64;                 // one nvfp4 block
    constexpr int NWARP = WARPS_M*WARPS_N;
    constexpr int THREADS = NWARP*32;
    constexpr int WM = BM/WARPS_M, WN = BN/WARPS_N;
    constexpr int MF = WM/16, NF = WN/8;

    constexpr int AN  = BK/2;                 // bf16 pairs per A smem row (nv_bfloat162)
    constexpr int ASTR = AN + 4;              // padded A smem row stride, in nv_bfloat162
    constexpr int SZ_A   = BM*ASTR;           // nv_bfloat162 per stage
    constexpr int SZ_WQ  = BN*8;              // u32 per stage (32 qs bytes/row)
    constexpr int SZ_WD  = BN;                // u32 per stage (4 scale bytes/row)

    extern __shared__ uint32_t smem_u32[];
    // Layout per stage: [A as u32 = nv_bfloat162][Wq u32][Wd u32]
    constexpr int STAGE_U32 = SZ_A + SZ_WQ + SZ_WD;
    nv_bfloat162 * sA[STAGES];
    uint32_t     * sWq[STAGES];
    uint32_t     * sWd[STAGES];
#pragma unroll
    for (int s = 0; s < STAGES; s++) {
        uint32_t * base = smem_u32 + s*STAGE_U32;
        sA[s]  = (nv_bfloat162 *) base;
        sWq[s] = base + SZ_A;
        sWd[s] = base + SZ_A + SZ_WQ;
    }

    // mma.cuh's tile ops (load_ldmatrix / mma / tile::get_i/get_j) use threadIdx.x AS THE WARP LANE,
    // so the block MUST be 2D (32, NWARP): threadIdx.x = lane (0..31), threadIdx.y = warp.
    const int lane = threadIdx.x;            // 0..31
    const int warp = threadIdx.y;            // 0..NWARP-1
    const int tid  = warp*32 + lane;         // linear id for the cp.async strided copies
    const int wrow = warp / WARPS_N, wcol = warp % WARPS_N;

    const w4a16_tile_desc tile_desc = g_tiles[blockIdx.y];
    const int    e        = tile_desc.expert;
    const int    row0     = tile_desc.row0;
    const int    rcount   = tile_desc.rows;
    const int    blockCol = blockIdx.x*BN;
    const int    Kb       = K/64;
    const block_nvfp4 * We = W0 + (int64_t) e*expert_stride_blocks; // expert e weight base

    tile_C acc[MF][NF];

    // async-load K-block `kt` into stage `st`
    auto load_tile = [&](int st, int kt) {
        // A: BM rows x BK bf16 = BM x AN nv_bfloat162 = BM x (BK/8) 16B chunks
        const int A_chunks = BM*(BK/8);
#pragma unroll 1
        for (int idx = tid; idx < A_chunks; idx += THREADS) {
            const int c = idx % (BK/8);          // 16B chunk in the row
            const int r = idx / (BK/8);          // row in tile
            const nv_bfloat16 * src = Abf + (int64_t)(row0 + r)*K + (int64_t)kt*BK + c*8;
            w4a16_cp_async16(((char *) sA[st]) + (r*ASTR + c*4)*sizeof(uint32_t), src);
        }
        // W qs: BN rows x 32 bytes = BN x 8 u32 (each block's qs at byte offset 4)
#pragma unroll 1
        for (int idx = tid; idx < BN*8; idx += THREADS) {
            const int w = idx & 7;               // u32 word in the 32-byte qs
            const int r = idx >> 3;              // row in tile
            const block_nvfp4 * blk = We + (int64_t)(blockCol + r)*Kb + kt;
            const char * src = ((const char *) blk) + 4 /*d[4]*/ + w*4;
            w4a16_cp_async4(&sWq[st][r*8 + w], src);
        }
        // W scales: BN rows x 4 bytes (one u32 each, the block's d[4] at byte offset 0)
#pragma unroll 1
        for (int r = tid; r < BN; r += THREADS) {
            const block_nvfp4 * blk = We + (int64_t)(blockCol + r)*Kb + kt;
            w4a16_cp_async4(&sWd[st][r], (const char *) blk);
        }
    };

    // prologue
#pragma unroll
    for (int s = 0; s < STAGES-1; s++) { if (s < Kb) load_tile(s, s); w4a16_cp_commit(); }

    for (int kt = 0; kt < Kb; kt++) {
        const int ld = kt + (STAGES-1);
        if (ld < Kb) load_tile(ld % STAGES, ld);
        w4a16_cp_commit();
        w4a16_cp_wait<STAGES-1>();
        __syncthreads();

        const int rs = kt % STAGES;
        const nv_bfloat162 * sAcur = sA[rs];
        const uint8_t      * sWqb  = (const uint8_t *) sWq[rs];   // BN rows x 32 bytes
        const uint32_t     * sWdw  = sWd[rs];                     // BN rows x 1 u32 (4 scale bytes)

#pragma unroll
        for (int kk = 0; kk < BK/16; kk++) {           // 4 m16n8k16 sub-steps per 64-block
            const int sub = kk;                        // sub-block (0..3): selects scale + nibble half
            // A fragments via ldmatrix (bf16)
            tile_A A_frag[MF];
#pragma unroll
            for (int mi = 0; mi < MF; mi++) {
                const int rb = wrow*WM + mi*16;
                load_ldmatrix(A_frag[mi], sAcur + rb*ASTR + kk*8, ASTR);
            }
            // B fragments: in-register FP4->bf16 dequant (correct-by-construction via tile get_i/get_j)
            tile_B B_frag[NF];
            const int n_local = lane >> 2;             // tile_B::get_i (row N, 0..7)
            const int jc      = lane & 3;              // lane%4
            const int qbyte   = sub*8 + 2*jc;          // qs byte index for this lane within the block
#pragma unroll
            for (int ni = 0; ni < NF; ni++) {
                const int nrow = wcol*WN + ni*8 + n_local;       // col within BN tile [0,BN)
                const uint8_t * qsb = sWqb + nrow*32;            // this row's 32 qs bytes
                const uint8_t   b0  = qsb[qbyte];
                const uint8_t   b1  = qsb[qbyte + 1];
                const float     sc  = ggml_cuda_ue4m3_to_fp32(((const uint8_t *) &sWdw[nrow])[sub]);
                // x[0]: low nibbles (k = 2jc, 2jc+1)
                B_frag[ni].x[0].x = __float2bfloat16(sc * (float) kvalues_mxfp4[b0 & 0x0F]);
                B_frag[ni].x[0].y = __float2bfloat16(sc * (float) kvalues_mxfp4[b1 & 0x0F]);
                // x[1]: high nibbles (k = 8+2jc, 9+2jc)
                B_frag[ni].x[1].x = __float2bfloat16(sc * (float) kvalues_mxfp4[b0 >> 4]);
                B_frag[ni].x[1].y = __float2bfloat16(sc * (float) kvalues_mxfp4[b1 >> 4]);
            }
#pragma unroll
            for (int mi = 0; mi < MF; mi++)
#pragma unroll
                for (int ni = 0; ni < NF; ni++)
                    mma(acc[mi][ni], A_frag[mi], B_frag[ni]);
        }
        __syncthreads();
    }

    // write back (mask the ragged per-expert row tail)
#pragma unroll
    for (int mi = 0; mi < MF; mi++)
#pragma unroll
        for (int ni = 0; ni < NF; ni++) {
            const int orow = wrow*WM + mi*16;
            const int ocol = blockCol + wcol*WN + ni*8;
#pragma unroll
            for (int l = 0; l < acc[mi][ni].ne; l++) {
                const int lr = orow + acc[mi][ni].get_i(l);   // local row within tile
                const int nc = ocol + acc[mi][ni].get_j(l);   // global col
                if (lr < rcount && nc < N) {
                    C[(int64_t)(row0 + lr)*N + nc] = acc[mi][ni].x[l];
                }
            }
        }
#else
    GGML_UNUSED(Abf); GGML_UNUSED(W0); GGML_UNUSED(C);
    GGML_UNUSED(g_tiles);
    GGML_UNUSED(N); GGML_UNUSED(K); GGML_UNUSED(expert_stride_blocks);
    NO_DEVICE_CODE;
#endif // AMPERE_MMA_AVAILABLE && CP_ASYNC_AVAILABLE
}

// ===========================================================================
// host integration
// ===========================================================================

bool ggml_cuda_w4a16_moe_grouped_should_engage(
        const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc) {
    if (src0->type != GGML_TYPE_NVFP4) {
        return false;
    }
    if (!blackwell_mma_available(cc)) {
        return false;
    }
    if (!ggml_cuda_w4a16_prefill_enabled()) {
        return false;                          // default-off == stock
    }
    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    // ne12 = total tokens (aggregate prefill M); only LARGE M (prefill), never decode.
    if (src1->ne[2] <= ggml_cuda_w4a16_prefill_m()) {
        return false;
    }
    const int64_t K = src0->ne[0];
    const int64_t N = src0->ne[1];
    if (N % 128 != 0 || K % 64 != 0) {
        return false;                          // tile constraints; else fall back to per-expert/MMQ
    }
    return true;
}

static bool ggml_cuda_w4a16_use_base_shape() {
    static const bool use_base = [] {
        const char * e = getenv("LLAMA_W4A16_SHAPE");
        if (e == nullptr || e[0] == '\0' || strcmp(e, "default") == 0 || strcmp(e, "bm32") == 0 || strcmp(e, "32x128") == 0) {
            return false;
        }
        if (strcmp(e, "base") == 0 || strcmp(e, "64x128") == 0) {
            return true;
        }
        fprintf(stderr, "[w4a16] unknown LLAMA_W4A16_SHAPE=%s, using default bm32\n", e);
        return false;
    }();
    return use_base;
}

template<int BM, int BN, int WARPS_M, int WARPS_N, int STAGES>
static void ggml_cuda_mul_mat_id_w4a16_grouped_impl(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const float * src1_sorted,
        float * dst_sorted,
        const int * tokens_per_expert,
        int64_t n_experts, int64_t K, int64_t N,
        cudaStream_t stream) {
    GGML_ASSERT(src0->type == GGML_TYPE_NVFP4);
    GGML_ASSERT(N % BN == 0 && K % 64 == 0);

    // host: build the per-M-tile expert map (ragged, no tile crosses an expert boundary)
    int64_t total_rows = 0;
    for (int64_t e = 0; e < n_experts; e++) {
        total_rows += tokens_per_expert[e];
    }
    if (total_rows == 0) {
        return;
    }

    std::vector<w4a16_tile_desc> h_tiles;
    int64_t row = 0;
    for (int64_t e = 0; e < n_experts; e++) {
        const int t = tokens_per_expert[e];
        for (int off = 0; off < t; off += BM) {
            h_tiles.push_back({
                (int) e,
                (int) (row + off),
                (int) std::min(BM, t - off),
                0,
            });
        }
        row += t;
    }
    const int n_tiles = (int) h_tiles.size();

    if (getenv("LLAMA_W4A16_DEBUG")) {
        int max_tpe = 0, multi = 0;
        for (int64_t e = 0; e < n_experts; e++) {
            if (tokens_per_expert[e] > max_tpe) max_tpe = tokens_per_expert[e];
            if (tokens_per_expert[e] > BM) multi++;
        }
        fprintf(stderr, "[w4a16] engaged: total_rows=%lld n_experts=%lld K=%lld N=%lld n_tiles=%d max_tpe=%d multi_tile_experts=%d\n",
                (long long) total_rows, (long long) n_experts, (long long) K, (long long) N, n_tiles, max_tpe, multi);
        fprintf(stderr, "[w4a16] shape: BM=%d BN=%d WARPS_M=%d WARPS_N=%d STAGES=%d\n",
                BM, BN, WARPS_M, WARPS_N, STAGES);
    }

    // device: packed tile map; one pageable H2D copy instead of three tiny copies
    ggml_cuda_pool_alloc<w4a16_tile_desc> d_tiles(ctx.pool(), n_tiles);
    CUDA_CHECK(cudaMemcpyAsync(d_tiles.ptr, h_tiles.data(), n_tiles*sizeof(w4a16_tile_desc), cudaMemcpyHostToDevice, stream));

    // activations: f32 -> bf16 (cheap cast, NO act-quant), zero-padded so every tile's BM-row read
    // stays in-bounds. A tile's row0 is generally NOT BM-aligned (experts start mid-buffer), and a
    // tile can begin as late as total_rows-1, so it can read up to total_rows-1+BM; add a full BM of
    // zero headroom on top of the BM-rounded length to cover that worst case.
    const int64_t pad_rows = (((total_rows + BM - 1) / BM) + 1) * BM;
    ggml_cuda_pool_alloc<nv_bfloat16> Abf(ctx.pool(), (size_t) pad_rows * K);
    {
        const int64_t n = total_rows * K;
        const int64_t npad = pad_rows * K;
        const int threads = 256;
        const int64_t grid = (npad + threads - 1) / threads;
        w4a16_cast_act_f32_bf16<<<grid, threads, 0, stream>>>(src1_sorted, Abf.get(), n, npad);
        CUDA_CHECK(cudaGetLastError());
    }

    const int64_t expert_stride_blocks = (int64_t) (src0->nb[2] / sizeof(block_nvfp4));

    auto kern = w4a16_grouped_kernel<BM, BN, WARPS_M, WARPS_N, STAGES>;
    constexpr int STAGE_U32 = BM*((64/2) + 4) + BN*8 + BN;
    const int smem_bytes = STAGES * STAGE_U32 * (int) sizeof(uint32_t);
    CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));

    dim3 grid((unsigned) (N / BN), (unsigned) n_tiles);
    dim3 block(32, WARPS_M*WARPS_N);   // 2D: threadIdx.x = warp lane, threadIdx.y = warp
    kern<<<grid, block, smem_bytes, stream>>>(
        Abf.get(), (const block_nvfp4 *) src0->data, dst_sorted,
        d_tiles.ptr,
        (int) N, (int) K, expert_stride_blocks);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_mul_mat_id_w4a16_grouped(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const float * src1_sorted,
        float * dst_sorted,
        const int * tokens_per_expert,
        int64_t n_experts, int64_t K, int64_t N,
        cudaStream_t stream) {
    if (ggml_cuda_w4a16_use_base_shape()) {
        ggml_cuda_mul_mat_id_w4a16_grouped_impl<64, 128, 2, 4, 2>(
                ctx, src0, src1_sorted, dst_sorted, tokens_per_expert, n_experts, K, N, stream);
    } else {
        ggml_cuda_mul_mat_id_w4a16_grouped_impl<32, 128, 1, 4, 2>(
                ctx, src0, src1_sorted, dst_sorted, tokens_per_expert, n_experts, K, N, stream);
    }
}
