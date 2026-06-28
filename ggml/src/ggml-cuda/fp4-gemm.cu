#include "fp4-gemm.cuh"

#include <cfloat>
#include <cstdint>
#include <cstdlib>

// ===========================================================================
// [paged patch 0034] Native NVFP4 (W4A4) large-M GEMM. See fp4-gemm.cuh.
//
// The GEMM kernel, the m16n8k64 block-scale OMMA wrapper, the cp.async helpers and
// the layout-split kernel are the VERIFIED PoC (fp4_gemm_w4a4_opt.cu, NMSE=0) copied
// verbatim - do not "tidy" the index math, it is the load-bearing correctness.
// ===========================================================================

#define FP4_QK 64           // == QK_NVFP4
#define FP4_SAW 8           // u32 per nvfp4 block qs (32 bytes)

#ifndef LLAMA_FP4_PREFILL_M
#define LLAMA_FP4_PREFILL_M 0
#endif // LLAMA_FP4_PREFILL_M

static int64_t ggml_cuda_fp4_prefill_m() {
    static const int64_t m = [] {
        const char * e = getenv("LLAMA_FP4_PREFILL_M");
        return e != nullptr ? (int64_t) atoll(e) : (int64_t) LLAMA_FP4_PREFILL_M;
    }();
    return m;
}

// ---- layout split: block_nvfp4[R*Kb] -> qs codes [R*Kb*8 u32] + scales [R*Kb u32] ----
// Same fp4 codes & e4m3 scale bytes as the GGUF, restored into two contiguous,
// 16B-friendly arrays so the kernel's cp.async copies are coalesced. (PoC verbatim.)
static __global__ void fp4_split_layout(
        const block_nvfp4 * __restrict__ X, uint32_t * __restrict__ Q, uint32_t * __restrict__ S,
        int R, int Kb) {
    const int64_t b   = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t tot = (int64_t) R * Kb;
    if (b >= tot) {
        return;
    }
    const block_nvfp4 & blk = X[b];
    const uint32_t * q = (const uint32_t *) blk.qs;
    uint32_t * dq = &Q[b * 8];
#pragma unroll
    for (int w = 0; w < 8; w++) {
        dq[w] = q[w];
    }
    S[b] = *(const uint32_t *) blk.d;
}

// ---- activation quantizer: f32 [M_real x K] -> split NVFP4 (Aq codes + As scales) ----
// Uses the SAME math as quantize_mmq_nvfp4 (quantize.cu): e4m3 scale = ue4m3(amax/6)
// with the +/-2 code search, ggml_cuda_float_to_fp4_e2m1 for the nibbles, so the
// activation codes are identical to the shipped FP4-MMQ path. Packs into the PoC
// block layout (qs[s*8+j] = code(e[j]) | code(e[j+8])<<4) expected by the kernel's
// ldmatrix A-operand load. One thread per (row, kb, sub-block).
static __global__ void fp4_quantize_act_split(
        const float * __restrict__ x, uint32_t * __restrict__ Aq, uint32_t * __restrict__ As,
        int M_real, int K, int Kb) {
#ifdef BLACKWELL_MMA_AVAILABLE
    const int64_t tot = (int64_t) M_real * Kb * 4; // 4 sub-blocks per 64-element block
    const int64_t t   = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= tot) {
        return;
    }
    const int     sub = (int) (t & 3);
    const int64_t rb  = t >> 2;            // row*Kb + kb
    const int     kb  = (int) (rb % Kb);
    const int64_t row = rb / Kb;

    const float * v16 = x + row * (int64_t) K + (int64_t) kb * FP4_QK + sub * 16;
    float vals[16];
    float amax = 0.0f;
#pragma unroll
    for (int k = 0; k < 16; k++) {
        const float vv = v16[k];
        vals[k] = vv;
        amax = fmaxf(amax, fabsf(vv));
    }

    static constexpr int test_offsets[5] = { 0, -1, 1, -2, 2 };
    const int first_fp8_code = (int) ggml_cuda_fp32_to_ue4m3(amax / 6.0f);

    float   best_err = FLT_MAX;
    uint8_t fp8_code = 0;
    float   subblock_scale = 0.0f;
#pragma unroll
    for (int i = 0; i < 5; i++) {
        const int test_code = first_fp8_code + test_offsets[i];
        if (test_code < 0 || test_code > 0x7e) {
            continue;
        }
        const uint8_t code           = (uint8_t) test_code;
        const float   test_scale     = ggml_cuda_ue4m3_to_fp32(code);
        const float   test_inv_scale = test_scale > 0.0f ? 0.5f / test_scale : 0.0f;
        float cur_err = 0.0f;
#pragma unroll
        for (int k = 0; k < 16; k++) {
            const uint8_t q        = ggml_cuda_float_to_fp4_e2m1(vals[k], test_inv_scale);
            const float   err_diff = fabsf(vals[k]) - fabsf((float) kvalues_mxfp4[q & 0x7]) * test_scale;
            cur_err = fmaf(err_diff, err_diff, cur_err);
        }
        if (cur_err < best_err) {
            best_err       = cur_err;
            fp8_code       = code;
            subblock_scale = test_scale;
        }
    }
    const float inv_scale = subblock_scale > 0.0f ? 0.5f / subblock_scale : 0.0f;

    // PoC packing: qs[s*8+j] = code(e[j]) | code(e[j+8])<<4 -> two u32 words per sub-block.
    uint32_t w0 = 0, w1 = 0;
#pragma unroll
    for (int j = 0; j < 4; j++) {
        const uint32_t lo = ggml_cuda_float_to_fp4_e2m1(vals[j],     inv_scale);
        const uint32_t hi = ggml_cuda_float_to_fp4_e2m1(vals[j + 8], inv_scale);
        w0 |= ((lo | (hi << 4)) & 0xff) << (8 * j);
    }
#pragma unroll
    for (int j = 0; j < 4; j++) {
        const uint32_t lo = ggml_cuda_float_to_fp4_e2m1(vals[j + 4],  inv_scale);
        const uint32_t hi = ggml_cuda_float_to_fp4_e2m1(vals[j + 12], inv_scale);
        w1 |= ((lo | (hi << 4)) & 0xff) << (8 * j);
    }

    const int64_t blk = row * (int64_t) Kb + kb;
    Aq[blk * 8 + sub * 2 + 0] = w0;
    Aq[blk * 8 + sub * 2 + 1] = w1;
    reinterpret_cast<uint8_t *>(As + blk)[sub] = fp8_code;
#else
    GGML_UNUSED(x); GGML_UNUSED(Aq); GGML_UNUSED(As);
    GGML_UNUSED(M_real); GGML_UNUSED(K); GGML_UNUSED(Kb);
    NO_DEVICE_CODE;
#endif // BLACKWELL_MMA_AVAILABLE
}

// ---- native FP4 block-scale OMMA wrapper (PoC verbatim) ----
static __device__ __forceinline__ void fp4_mma(
        float d[4], const uint32_t a[4], const uint32_t b[2], uint32_t as, uint32_t bs) {
#ifdef BLACKWELL_MMA_AVAILABLE
    asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},%10,{0,0},%11,{0,0};"
      : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
      : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]),"r"(as),"r"(bs));
#else
    GGML_UNUSED(d); GGML_UNUSED(a); GGML_UNUSED(b); GGML_UNUSED(as); GGML_UNUSED(bs);
    NO_DEVICE_CODE;
#endif // BLACKWELL_MMA_AVAILABLE
}

// ---- cp.async helpers (PoC verbatim) ----
static __device__ __forceinline__ void fp4_cp_async16(void * smem, const void * gmem) {
#ifdef CP_ASYNC_AVAILABLE
    unsigned s = (unsigned) __cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0],[%1],16;\n" :: "r"(s), "l"(gmem));
#else
    GGML_UNUSED(smem); GGML_UNUSED(gmem); NO_DEVICE_CODE;
#endif // CP_ASYNC_AVAILABLE
}
template<int B>
static __device__ __forceinline__ void fp4_cp_async_small(void * smem, const void * gmem) {
#ifdef CP_ASYNC_AVAILABLE
    unsigned s = (unsigned) __cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0],[%1],%2;\n" :: "r"(s), "l"(gmem), "n"(B));
#else
    GGML_UNUSED(smem); GGML_UNUSED(gmem); NO_DEVICE_CODE;
#endif // CP_ASYNC_AVAILABLE
}
static __device__ __forceinline__ void fp4_cp_commit() {
#ifdef CP_ASYNC_AVAILABLE
    asm volatile("cp.async.commit_group;\n" ::);
#else
    NO_DEVICE_CODE;
#endif // CP_ASYNC_AVAILABLE
}
template<int N>
static __device__ __forceinline__ void fp4_cp_wait() {
#ifdef CP_ASYNC_AVAILABLE
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
#else
    NO_DEVICE_CODE;
#endif // CP_ASYNC_AVAILABLE
}

// ---------------------------------------------------------------------------
// Optimized native FP4 GEMM (PoC verbatim). C[M,N] = A_fp4[M,K] @ W_fp4[N,K]^T
// inputs are layout-split:  Aq[M*Kb*8], As[M*Kb], Wq[N*Kb*8], Ws[N*Kb]
// Tile BM x BN, K-step = KBLK nvfp4 blocks (BK = 64*KBLK), STAGES-deep pipeline,
// PAD u32 padding per smem row to defeat bank conflicts.
// ---------------------------------------------------------------------------
template<int BM,int BN,int WARPS_M,int WARPS_N,int KBLK,int STAGES,int PAD>
__launch_bounds__(WARPS_M*WARPS_N*32,1)
static __global__ void fp4_opt_kernel(
        const uint32_t * __restrict__ Aq, const uint32_t * __restrict__ As,
        const uint32_t * __restrict__ Wq, const uint32_t * __restrict__ Ws,
        float * __restrict__ C, int M, int N, int K) {
#ifdef BLACKWELL_MMA_AVAILABLE
    constexpr int NWARP=WARPS_M*WARPS_N;
    constexpr int THREADS=NWARP*32;
    constexpr int WM=BM/WARPS_M, WN=BN/WARPS_N;
    constexpr int MF=WM/16, NF=WN/8;
    constexpr int SAW=8;                 // u32 per block (qs)
    constexpr int ARS=KBLK*SAW+PAD;      // A smem row stride (u32)
    constexpr int WRS=KBLK*SAW+PAD;      // W smem row stride (u32)

    extern __shared__ uint32_t smem[];
    // per-stage slabs
    constexpr int SZ_AQ=BM*ARS, SZ_AS=BM*KBLK, SZ_WQ=BN*WRS, SZ_WS=BN*KBLK;
    constexpr int STAGE_SZ=SZ_AQ+SZ_AS+SZ_WQ+SZ_WS;
    uint32_t* sAq[STAGES]; uint32_t* sAs[STAGES]; uint32_t* sWq[STAGES]; uint32_t* sWs[STAGES];
#pragma unroll
    for(int s=0;s<STAGES;s++){
        uint32_t* base=smem+s*STAGE_SZ;
        sAq[s]=base; sAs[s]=base+SZ_AQ; sWq[s]=base+SZ_AQ+SZ_AS; sWs[s]=base+SZ_AQ+SZ_AS+SZ_WQ;
    }

    const int tid=threadIdx.x, warp=tid>>5, lane=tid&31;
    const int wrow=warp/WARPS_N, wcol=warp%WARPS_N;
    const int grp=lane>>2, tig=lane&3;
    const int tidxA = lane/4 + (lane%2)*8;
    const int tidxB = lane/4;
    const int blockRow=blockIdx.y*BM, blockCol=blockIdx.x*BN;
    const int Kb=K/64;
    const int numK=Kb/KBLK;

    float acc[MF][NF][4];
#pragma unroll
    for(int i=0;i<MF;i++)for(int j=0;j<NF;j++)for(int r=0;r<4;r++)acc[i][j][r]=0;

    // async-load k-tile `kt` into stage `st`
    auto load_tile=[&](int st,int kt){
        const int kb0=kt*KBLK;
        // A qs: BM*KBLK blocks, 2x 16B chunks each
#pragma unroll 1
        for(int idx=tid; idx<BM*KBLK*2; idx+=THREADS){
            int chunk=idx&1, blk=idx>>1;
            int r=blk/KBLK, kb=blk%KBLK;
            const uint32_t* src=&Aq[((size_t)(blockRow+r)*Kb + kb0+kb)*SAW + chunk*4];
            fp4_cp_async16(&sAq[st][r*ARS + kb*SAW + chunk*4], src);
        }
        // W qs
#pragma unroll 1
        for(int idx=tid; idx<BN*KBLK*2; idx+=THREADS){
            int chunk=idx&1, blk=idx>>1;
            int r=blk/KBLK, kb=blk%KBLK;
            const uint32_t* src=&Wq[((size_t)(blockCol+r)*Kb + kb0+kb)*SAW + chunk*4];
            fp4_cp_async16(&sWq[st][r*WRS + kb*SAW + chunk*4], src);
        }
        // A scales: BM rows, KBLK contiguous u32 each
#pragma unroll 1
        for(int r=tid; r<BM; r+=THREADS){
            const uint32_t* src=&As[(size_t)(blockRow+r)*Kb + kb0];
            uint32_t* dst=&sAs[st][r*KBLK];
            if(KBLK==4) fp4_cp_async16(dst,src);
            else if(KBLK==2) fp4_cp_async_small<8>(dst,src);
            else fp4_cp_async_small<4>(dst,src);
        }
        // W scales
#pragma unroll 1
        for(int r=tid; r<BN; r+=THREADS){
            const uint32_t* src=&Ws[(size_t)(blockCol+r)*Kb + kb0];
            uint32_t* dst=&sWs[st][r*KBLK];
            if(KBLK==4) fp4_cp_async16(dst,src);
            else if(KBLK==2) fp4_cp_async_small<8>(dst,src);
            else fp4_cp_async_small<4>(dst,src);
        }
    };

    // prologue: issue STAGES-1 tiles (tiles 0..STAGES-2 into stages 0..STAGES-2)
#pragma unroll
    for(int s=0;s<STAGES-1;s++){ if(s<numK) load_tile(s,s); fp4_cp_commit(); }

    for(int kt=0; kt<numK; kt++){
        // prefetch tile kt+STAGES-1 into its stage (overlaps this iter's compute)
        int ld=kt+(STAGES-1);
        if(ld<numK) load_tile(ld%STAGES,ld);
        fp4_cp_commit();
        // wait until tile kt has landed (leave STAGES-1 prefetches in flight)
        fp4_cp_wait<STAGES-1>();
        __syncthreads();

        const int rs=kt%STAGES;
#pragma unroll
        for(int kb=0; kb<KBLK; kb++){
            // A fragments via ldmatrix (PRESERVED layout)
            uint32_t af[MF][4]; uint32_t asc[MF];
#pragma unroll
            for(int mi=0; mi<MF; mi++){
                int rb=wrow*WM+mi*16;
                const uint32_t* base=&sAq[rs][rb*ARS + kb*SAW];
                const uint32_t* xs = base + (lane%16)*ARS + (lane/16)*4;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3},[%4];"
                    : "=r"(af[mi][0]),"=r"(af[mi][1]),"=r"(af[mi][2]),"=r"(af[mi][3])
                    : "l"(xs));
                asc[mi]=sAs[rs][(rb+tidxA)*KBLK+kb];
            }
            // B fragments (PRESERVED manual gather), padded row stride
            uint32_t bf[NF][2]; uint32_t bsc[NF];
#pragma unroll
            for(int ni=0; ni<NF; ni++){
                int nb=wcol*WN+ni*8;
                const uint32_t* base=&sWq[rs][nb*WRS + kb*SAW];
#pragma unroll
                for(int l=0;l<2;l++){
                    int gi=grp, gj=l*4+tig;
                    bf[ni][l]=base[gi*WRS + gj];
                }
                bsc[ni]=sWs[rs][(nb+tidxB)*KBLK+kb];
            }
#pragma unroll
            for(int mi=0;mi<MF;mi++)
#pragma unroll
                for(int ni=0;ni<NF;ni++)
                    fp4_mma(acc[mi][ni], af[mi], bf[ni], asc[mi], bsc[ni]);
        }
        // ensure all warps finished reading stage rs before it is reused by a
        // future prefetch (the stage is overwritten at iter kt+1's prefetch).
        __syncthreads();
    }

#pragma unroll
    for(int mi=0;mi<MF;mi++)
#pragma unroll
        for(int ni=0;ni<NF;ni++){
            int orb=blockRow+wrow*WM+mi*16, ocb=blockCol+wcol*WN+ni*8;
            float* d=acc[mi][ni];
            C[(size_t)(orb+grp)*N+ocb+2*tig]    =d[0];
            C[(size_t)(orb+grp)*N+ocb+2*tig+1]  =d[1];
            C[(size_t)(orb+grp+8)*N+ocb+2*tig]  =d[2];
            C[(size_t)(orb+grp+8)*N+ocb+2*tig+1]=d[3];
        }
#else
    GGML_UNUSED(Aq); GGML_UNUSED(As); GGML_UNUSED(Wq); GGML_UNUSED(Ws);
    GGML_UNUSED(C); GGML_UNUSED(M); GGML_UNUSED(N); GGML_UNUSED(K);
    NO_DEVICE_CODE;
#endif // BLACKWELL_MMA_AVAILABLE
}

// ===========================================================================
// ggml integration
// ===========================================================================

bool ggml_cuda_fp4_prefill_should_engage(
        const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc) {
    if (src0->type != GGML_TYPE_NVFP4) {
        return false;
    }
    if (!blackwell_mma_available(cc)) {
        return false;
    }
    const int64_t thr = ggml_cuda_fp4_prefill_m();
    if (thr <= 0) {
        return false;                       // default-off == stock; decode/small-M untouched
    }
    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (src1->ne[1] <= thr) {
        return false;                       // M = src1->ne[1]; only LARGE M (prefill)
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        return false;
    }
    if (ggml_is_transposed(src0) || ggml_is_transposed(src1)) {
        return false;
    }
    // 2D only (a single weight matrix; per-expert MoE slices set ne[2]=ne[3]=1).
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) {
        return false;
    }
    const int64_t K = src0->ne[0];
    const int64_t N = src0->ne[1];
    if (N % 128 != 0 || K % 256 != 0) {
        return false;                       // tile constraints; otherwise fall back to MMQ
    }
    return true;
}

void ggml_cuda_mul_mat_fp4_large_m(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_NVFP4);
    GGML_ASSERT(src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);

    const int64_t K  = src0->ne[0];
    const int64_t N  = src0->ne[1];
    const int64_t M  = src1->ne[1];
    const int64_t Kb = K / FP4_QK;
    GGML_ASSERT(K % 256 == 0 && N % 128 == 0);

    cudaStream_t stream = ctx.stream();

    constexpr int BM = 128, BN = 128, WM = 4, WN = 2, KBLK = 4, STAGES = 2, PAD = 4;
    const int64_t Mpad = ((M + BM - 1) / BM) * BM;

    ggml_cuda_pool_alloc<uint32_t> Wq(ctx.pool(), (size_t) N * Kb * 8);
    ggml_cuda_pool_alloc<uint32_t> Ws(ctx.pool(), (size_t) N * Kb);
    ggml_cuda_pool_alloc<uint32_t> Aq(ctx.pool(), (size_t) Mpad * Kb * 8);
    ggml_cuda_pool_alloc<uint32_t> As(ctx.pool(), (size_t) Mpad * Kb);

    // Zero the scales of the padded A-rows (M..Mpad) so they contribute 0 (scale 0 ->
    // the OMMA's per-block scale is 0). The padded qs may stay uninitialized.
    if (Mpad > M) {
        CUDA_CHECK(cudaMemsetAsync(As.get() + (size_t) M * Kb, 0,
                                   (size_t) (Mpad - M) * Kb * sizeof(uint32_t), stream));
    }

    // split weights (GGUF block_nvfp4 -> Wq/Ws)
    {
        const int64_t tot     = N * Kb;
        const int     threads = 256;
        const int64_t grid    = (tot + threads - 1) / threads;
        fp4_split_layout<<<grid, threads, 0, stream>>>(
            (const block_nvfp4 *) src0->data, Wq.get(), Ws.get(), (int) N, (int) Kb);
        CUDA_CHECK(cudaGetLastError());
    }
    // quantize + split activations (real rows only)
    {
        const int64_t tot     = M * Kb * 4;
        const int     threads = 256;
        const int64_t grid    = (tot + threads - 1) / threads;
        fp4_quantize_act_split<<<grid, threads, 0, stream>>>(
            (const float *) src1->data, Aq.get(), As.get(), (int) M, (int) K, (int) Kb);
        CUDA_CHECK(cudaGetLastError());
    }

    // Output: write the (Mpad x N) result straight into dst when M is tile-aligned,
    // otherwise into a temp and copy back the first M rows (C is row-major C[m*N+n]).
    float * Cout = (float *) dst->data;
    ggml_cuda_pool_alloc<float> Ctmp(ctx.pool());
    if (Mpad > M) {
        Cout = Ctmp.alloc((size_t) Mpad * N);
    }

    auto kern = fp4_opt_kernel<BM, BN, WM, WN, KBLK, STAGES, PAD>;
    constexpr int SZ_AQ = BM * (KBLK * 8 + PAD), SZ_AS = BM * KBLK;
    constexpr int SZ_WQ = BN * (KBLK * 8 + PAD), SZ_WS = BN * KBLK;
    constexpr int STAGE_SZ = SZ_AQ + SZ_AS + SZ_WQ + SZ_WS;
    const int smem_bytes = STAGES * STAGE_SZ * (int) sizeof(uint32_t);
    CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));

    dim3 grid((unsigned) (N / BN), (unsigned) (Mpad / BM));
    dim3 block(WM * WN * 32);
    kern<<<grid, block, smem_bytes, stream>>>(
        Aq.get(), As.get(), Wq.get(), Ws.get(), Cout, (int) Mpad, (int) N, (int) K);
    CUDA_CHECK(cudaGetLastError());

    if (Mpad > M) {
        CUDA_CHECK(cudaMemcpyAsync(dst->data, Ctmp.get(), (size_t) M * N * sizeof(float),
                                   cudaMemcpyDeviceToDevice, stream));
    }
}
