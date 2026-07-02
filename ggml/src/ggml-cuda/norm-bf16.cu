#include "norm-bf16.cuh"

#include <cstring>

// [P1 bf16-stream] bf16-output variants of the residual-stream norms. Same reduction
// and FP order as the f32 kernels in norm.cu; only the final store is narrowed. Kept
// bit-faithful to the f32 norms up to the __float2bfloat16 store so the opt-in stream
// stays a pure dtype (KL-benign) change, not an algorithmic one.

// Output-store policy: identity for float, round-to-nearest bf16 for nv_bfloat16.
template <typename Tdst> struct bf16stream_store;
template <> struct bf16stream_store<float> {
    static __device__ __forceinline__ float store(float v) { return v; }
};
template <> struct bf16stream_store<nv_bfloat16> {
    static __device__ __forceinline__ nv_bfloat16 store(float v) { return __float2bfloat16(v); }
};

// ---------------------------------------------------------------------------
// plain rms_norm + weight multiply -> Tdst    (mirrors rms_norm_f32<do_multiply=true>)
// ---------------------------------------------------------------------------
template <int block_size, typename Tdst>
static __global__ void rms_norm_mul_out(const float * x,
                                        Tdst *        dst,
                                        const int     ncols,
                                        const int64_t stride_row,
                                        const int64_t stride_channel,
                                        const int64_t stride_sample,
                                        const float   eps,
                                        const float * mul,
                                        const int64_t mul_stride_row,
                                        const int64_t mul_stride_channel,
                                        const int64_t mul_stride_sample,
                                        const uint3   mul_ncols_packed,
                                        const uint3   mul_nrows_packed,
                                        const uint3   mul_nchannels_packed,
                                        const uint3   mul_nsamples_packed) {
    ggml_cuda_pdl_lc();
    const int nrows     = gridDim.x;
    const int nchannels = gridDim.y;

    const int row     = blockIdx.x;
    const int channel = blockIdx.y;
    const int sample  = blockIdx.z;
    const int tid     = threadIdx.x;

    x   += sample*stride_sample + channel*stride_channel + row*stride_row;
    dst += ((sample*nchannels + channel)*nrows + row)*ncols;

    {
        const uint32_t mul_row     = fastmodulo(row, mul_nrows_packed);
        const uint32_t mul_channel = fastmodulo(channel, mul_nchannels_packed);
        const uint32_t mul_sample  = fastmodulo(sample, mul_nsamples_packed);
        mul += mul_sample * mul_stride_sample + mul_channel * mul_stride_channel + mul_row * mul_stride_row;
    }

    float tmp = 0.0f;

    ggml_cuda_pdl_sync();
    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        tmp += xi * xi;
    }

    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

    const float mean  = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    for (int col = tid; col < ncols; col += block_size) {
        const int mul_col = fastmodulo(col, mul_ncols_packed);
        dst[col]          = bf16stream_store<Tdst>::store(scale * x[col] * mul[mul_col]);
    }
}

// ---------------------------------------------------------------------------
// 0042 residual-add + rms_norm + weight multiply -> f32 h_out + Tdst dst
// (mirrors rms_norm_pre_add_mul_f32<do_multiply=true>; h_out stays f32 so the next
//  residual add reads the same f32 residual stream)
// ---------------------------------------------------------------------------
template <int block_size, typename Tdst>
static __global__ void rms_norm_pre_add_mul_out(const float * a,
                                                const float * b,
                                                float *       h_out,
                                                Tdst *        dst,
                                                const int     ncols,
                                                const int64_t stride_row,
                                                const int64_t stride_channel,
                                                const int64_t stride_sample,
                                                const float   eps,
                                                const float * mul,
                                                const int64_t mul_stride_row,
                                                const int64_t mul_stride_channel,
                                                const int64_t mul_stride_sample,
                                                const uint3   mul_ncols_packed,
                                                const uint3   mul_nrows_packed,
                                                const uint3   mul_nchannels_packed,
                                                const uint3   mul_nsamples_packed) {
    ggml_cuda_pdl_lc();
    const int nrows     = gridDim.x;
    const int nchannels = gridDim.y;

    const int row     = blockIdx.x;
    const int channel = blockIdx.y;
    const int sample  = blockIdx.z;
    const int tid     = threadIdx.x;

    const int64_t row_offset = sample*stride_sample + channel*stride_channel + row*stride_row;
    a     += row_offset;
    b     += row_offset;
    h_out += row_offset;
    dst   += ((sample*nchannels + channel)*nrows + row)*ncols;

    {
        const uint32_t mul_row     = fastmodulo(row, mul_nrows_packed);
        const uint32_t mul_channel = fastmodulo(channel, mul_nchannels_packed);
        const uint32_t mul_sample  = fastmodulo(sample, mul_nsamples_packed);
        mul += mul_sample * mul_stride_sample + mul_channel * mul_stride_channel + mul_row * mul_stride_row;
    }

    float tmp = 0.0f;

    ggml_cuda_pdl_sync();
    for (int col = tid; col < ncols; col += block_size) {
        const float hi = a[col] + b[col];
        h_out[col] = hi;   // publish the f32 residual stream for the next add
        tmp += hi * hi;
    }

    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

    const float mean  = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    for (int col = tid; col < ncols; col += block_size) {
        const float hi      = h_out[col];
        const int   mul_col = fastmodulo(col, mul_ncols_packed);
        dst[col]            = bf16stream_store<Tdst>::store(scale * hi * mul[mul_col]);
    }
}

// ---------------------------------------------------------------------------
// 0044 gated-DeltaNet output norm  scale*x*w*silu(z) -> Tdst  (the P0 segment)
// ---------------------------------------------------------------------------
template <int block_size, typename Tdst>
static __global__ void rms_norm_gate_mul_out(const float * x,
                                             Tdst *        dst,
                                             const int     ncols,
                                             const int64_t stride_row,
                                             const int64_t stride_channel,
                                             const int64_t stride_sample,
                                             const float   eps,
                                             const float * mul,
                                             const int64_t mul_stride_row,
                                             const int64_t mul_stride_channel,
                                             const int64_t mul_stride_sample,
                                             const uint3   mul_ncols_packed,
                                             const uint3   mul_nrows_packed,
                                             const uint3   mul_nchannels_packed,
                                             const uint3   mul_nsamples_packed,
                                             const float * gate,
                                             const int64_t gate_stride_row,
                                             const int64_t gate_stride_channel,
                                             const int64_t gate_stride_sample,
                                             const uint3   gate_ncols_packed,
                                             const uint3   gate_nrows_packed,
                                             const uint3   gate_nchannels_packed,
                                             const uint3   gate_nsamples_packed) {
    ggml_cuda_pdl_lc();
    const int nrows     = gridDim.x;
    const int nchannels = gridDim.y;

    const int row     = blockIdx.x;
    const int channel = blockIdx.y;
    const int sample  = blockIdx.z;
    const int tid     = threadIdx.x;

    x   += sample*stride_sample + channel*stride_channel + row*stride_row;
    dst += ((sample*nchannels + channel)*nrows + row)*ncols;

    {
        const uint32_t mul_row     = fastmodulo(row, mul_nrows_packed);
        const uint32_t mul_channel = fastmodulo(channel, mul_nchannels_packed);
        const uint32_t mul_sample  = fastmodulo(sample, mul_nsamples_packed);
        mul += mul_sample * mul_stride_sample + mul_channel * mul_stride_channel + mul_row * mul_stride_row;
    }
    {
        const uint32_t gate_row     = fastmodulo(row, gate_nrows_packed);
        const uint32_t gate_channel = fastmodulo(channel, gate_nchannels_packed);
        const uint32_t gate_sample  = fastmodulo(sample, gate_nsamples_packed);
        gate += gate_sample * gate_stride_sample + gate_channel * gate_stride_channel + gate_row * gate_stride_row;
    }

    float tmp = 0.0f;

    ggml_cuda_pdl_sync();
    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        tmp += xi * xi;
    }

    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

    const float mean  = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    for (int col = tid; col < ncols; col += block_size) {
        const int   mul_col  = fastmodulo(col, mul_ncols_packed);
        const int   gate_col = fastmodulo(col, gate_ncols_packed);
        const float zi       = gate[gate_col];
        const float silu_z   = zi / (1.0f + expf(-zi));
        dst[col]             = bf16stream_store<Tdst>::store(scale * x[col] * mul[mul_col] * silu_z);
    }
}

// ===========================================================================
// launchers
// ===========================================================================
template <typename Tdst>
static void rms_norm_mul_out_cuda(const float * x, Tdst * dst,
                                  const int ncols, const int nrows, const int nchannels, const int nsamples,
                                  const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample,
                                  const float * mul,
                                  const int64_t mul_stride_row, const int64_t mul_stride_channel, const int64_t mul_stride_sample,
                                  const uint32_t mul_ncols, const uint32_t mul_nrows, const uint32_t mul_nchannels, const uint32_t mul_nsamples,
                                  const float eps, cudaStream_t stream) {
    const dim3 blocks_num(nrows, nchannels, nsamples);
    GGML_ASSERT(mul != nullptr);
    const uint3 mc = init_fastdiv_values(mul_ncols);
    const uint3 mr = init_fastdiv_values(mul_nrows);
    const uint3 mch = init_fastdiv_values(mul_nchannels);
    const uint3 ms = init_fastdiv_values(mul_nsamples);
    if (ncols < 1024) {
        const dim3 block_dims(256, 1, 1);
        const ggml_cuda_kernel_launch_params lp{blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float) : 0, stream};
        ggml_cuda_kernel_launch(rms_norm_mul_out<256, Tdst>, lp,
            x, dst, ncols, stride_row, stride_channel, stride_sample, eps,
            mul, mul_stride_row, mul_stride_channel, mul_stride_sample, mc, mr, mch, ms);
    } else {
        const dim3 block_dims(1024, 1, 1);
        const ggml_cuda_kernel_launch_params lp{blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float) : 0, stream};
        ggml_cuda_kernel_launch(rms_norm_mul_out<1024, Tdst>, lp,
            x, dst, ncols, stride_row, stride_channel, stride_sample, eps,
            mul, mul_stride_row, mul_stride_channel, mul_stride_sample, mc, mr, mch, ms);
    }
}

template <typename Tdst>
static void rms_norm_pre_add_mul_out_cuda(const float * a, const float * b, float * h_out, Tdst * dst,
                                          const int ncols, const int nrows, const int nchannels, const int nsamples,
                                          const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample,
                                          const float * mul,
                                          const int64_t mul_stride_row, const int64_t mul_stride_channel, const int64_t mul_stride_sample,
                                          const uint32_t mul_ncols, const uint32_t mul_nrows, const uint32_t mul_nchannels, const uint32_t mul_nsamples,
                                          const float eps, cudaStream_t stream) {
    const dim3 blocks_num(nrows, nchannels, nsamples);
    GGML_ASSERT(mul != nullptr);
    const uint3 mc = init_fastdiv_values(mul_ncols);
    const uint3 mr = init_fastdiv_values(mul_nrows);
    const uint3 mch = init_fastdiv_values(mul_nchannels);
    const uint3 ms = init_fastdiv_values(mul_nsamples);
    if (ncols < 1024) {
        const dim3 block_dims(256, 1, 1);
        const ggml_cuda_kernel_launch_params lp{blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float) : 0, stream};
        ggml_cuda_kernel_launch(rms_norm_pre_add_mul_out<256, Tdst>, lp,
            a, b, h_out, dst, ncols, stride_row, stride_channel, stride_sample, eps,
            mul, mul_stride_row, mul_stride_channel, mul_stride_sample, mc, mr, mch, ms);
    } else {
        const dim3 block_dims(1024, 1, 1);
        const ggml_cuda_kernel_launch_params lp{blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float) : 0, stream};
        ggml_cuda_kernel_launch(rms_norm_pre_add_mul_out<1024, Tdst>, lp,
            a, b, h_out, dst, ncols, stride_row, stride_channel, stride_sample, eps,
            mul, mul_stride_row, mul_stride_channel, mul_stride_sample, mc, mr, mch, ms);
    }
}

template <typename Tdst>
static void rms_norm_gate_mul_out_cuda(const float * x, Tdst * dst,
                                       const int ncols, const int nrows, const int nchannels, const int nsamples,
                                       const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample,
                                       const float * mul,
                                       const int64_t mul_stride_row, const int64_t mul_stride_channel, const int64_t mul_stride_sample,
                                       const uint32_t mul_ncols, const uint32_t mul_nrows, const uint32_t mul_nchannels, const uint32_t mul_nsamples,
                                       const float * gate,
                                       const int64_t gate_stride_row, const int64_t gate_stride_channel, const int64_t gate_stride_sample,
                                       const uint32_t gate_ncols, const uint32_t gate_nrows, const uint32_t gate_nchannels, const uint32_t gate_nsamples,
                                       const float eps, cudaStream_t stream) {
    const dim3 blocks_num(nrows, nchannels, nsamples);
    GGML_ASSERT(mul  != nullptr);
    GGML_ASSERT(gate != nullptr);
    const uint3 mc = init_fastdiv_values(mul_ncols);
    const uint3 mr = init_fastdiv_values(mul_nrows);
    const uint3 mch = init_fastdiv_values(mul_nchannels);
    const uint3 ms = init_fastdiv_values(mul_nsamples);
    const uint3 gc = init_fastdiv_values(gate_ncols);
    const uint3 gr = init_fastdiv_values(gate_nrows);
    const uint3 gch = init_fastdiv_values(gate_nchannels);
    const uint3 gs = init_fastdiv_values(gate_nsamples);
    if (ncols < 1024) {
        const dim3 block_dims(256, 1, 1);
        const ggml_cuda_kernel_launch_params lp{blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float) : 0, stream};
        ggml_cuda_kernel_launch(rms_norm_gate_mul_out<256, Tdst>, lp,
            x, dst, ncols, stride_row, stride_channel, stride_sample, eps,
            mul, mul_stride_row, mul_stride_channel, mul_stride_sample, mc, mr, mch, ms,
            gate, gate_stride_row, gate_stride_channel, gate_stride_sample, gc, gr, gch, gs);
    } else {
        const dim3 block_dims(1024, 1, 1);
        const ggml_cuda_kernel_launch_params lp{blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float) : 0, stream};
        ggml_cuda_kernel_launch(rms_norm_gate_mul_out<1024, Tdst>, lp,
            x, dst, ncols, stride_row, stride_channel, stride_sample, eps,
            mul, mul_stride_row, mul_stride_channel, mul_stride_sample, mc, mr, mch, ms,
            gate, gate_stride_row, gate_stride_channel, gate_stride_sample, gc, gr, gch, gs);
    }
}

// ===========================================================================
// host entries
// ===========================================================================
void ggml_cuda_rms_norm_mul_bf16out(ggml_backend_cuda_context & ctx,
                                    const ggml_tensor *         rms_norm_tensor,
                                    const ggml_tensor *         mul_tensor,
                                    void *                      dst_bf16) {
    const ggml_tensor * x_src   = rms_norm_tensor->src[0];
    const ggml_tensor * mul_src = (mul_tensor->src[0] == rms_norm_tensor) ? mul_tensor->src[1] : mul_tensor->src[0];
    GGML_ASSERT(mul_tensor->src[0] == rms_norm_tensor || mul_tensor->src[1] == rms_norm_tensor);

    float eps = 0.0f;
    memcpy(&eps, rms_norm_tensor->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    GGML_ASSERT(x_src->type           == GGML_TYPE_F32);
    GGML_ASSERT(mul_src->type         == GGML_TYPE_F32);
    GGML_ASSERT(rms_norm_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(mul_tensor->type      == GGML_TYPE_F32);

    const float * x_d   = (const float *) x_src->data;
    const float * mul_d = (const float *) mul_src->data;
    nv_bfloat16 * dst_d = (nv_bfloat16 *)  dst_bf16;
    cudaStream_t  stream = ctx.stream();

    const int64_t ne00 = rms_norm_tensor->ne[0];
    const int64_t ne01 = rms_norm_tensor->ne[1];
    const int64_t ne02 = rms_norm_tensor->ne[2];
    const int64_t ne03 = rms_norm_tensor->ne[3];

    const size_t ts0 = ggml_type_size(x_src->type);
    GGML_ASSERT(x_src->nb[0] == ts0);
    const int64_t s01 = x_src->nb[1] / ts0;
    const int64_t s02 = x_src->nb[2] / ts0;
    const int64_t s03 = x_src->nb[3] / ts0;

    const size_t ts_mul = ggml_type_size(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    const int64_t mul_s01 = mul_src->nb[1] / ts_mul;
    const int64_t mul_s02 = mul_src->nb[2] / ts_mul;
    const int64_t mul_s03 = mul_src->nb[3] / ts_mul;

    rms_norm_mul_out_cuda<nv_bfloat16>(x_d, dst_d,
        ne00, ne01, ne02, ne03, s01, s02, s03,
        mul_d, mul_s01, mul_s02, mul_s03,
        mul_src->ne[0], mul_src->ne[1], mul_src->ne[2], mul_src->ne[3],
        eps, stream);
}

void ggml_cuda_rms_norm_pre_add_mul_bf16out(ggml_backend_cuda_context & ctx,
                                            const ggml_tensor *         add_tensor,
                                            const ggml_tensor *         rms_norm_tensor,
                                            const ggml_tensor *         mul_tensor,
                                            void *                      dst_bf16) {
    GGML_ASSERT(rms_norm_tensor->src[0] == add_tensor);

    const ggml_tensor * a_src = add_tensor->src[0];
    const ggml_tensor * b_src = add_tensor->src[1];

    float eps = 0.0f;
    memcpy(&eps, rms_norm_tensor->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    const ggml_tensor * mul_src = (mul_tensor->src[0] == rms_norm_tensor) ? mul_tensor->src[1] : mul_tensor->src[0];
    GGML_ASSERT(mul_tensor->src[0] == rms_norm_tensor || mul_tensor->src[1] == rms_norm_tensor);

    GGML_ASSERT(a_src->type           == GGML_TYPE_F32);
    GGML_ASSERT(b_src->type           == GGML_TYPE_F32);
    GGML_ASSERT(add_tensor->type      == GGML_TYPE_F32);
    GGML_ASSERT(rms_norm_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(mul_tensor->type      == GGML_TYPE_F32);
    GGML_ASSERT(ggml_are_same_shape(a_src, b_src));

    const float * a_d   = (const float *) a_src->data;
    const float * b_d   = (const float *) b_src->data;
    float *       h_d   = (float *)       add_tensor->data;   // f32 residual stream
    const float * mul_d = (const float *) mul_src->data;
    nv_bfloat16 * dst_d = (nv_bfloat16 *)  dst_bf16;
    cudaStream_t  stream = ctx.stream();

    const int64_t ne00 = add_tensor->ne[0];
    const int64_t ne01 = add_tensor->ne[1];
    const int64_t ne02 = add_tensor->ne[2];
    const int64_t ne03 = add_tensor->ne[3];

    const size_t ts0 = ggml_type_size(a_src->type);
    GGML_ASSERT(a_src->nb[0] == ts0 && b_src->nb[0] == ts0);
    const int64_t s01 = a_src->nb[1] / ts0;
    const int64_t s02 = a_src->nb[2] / ts0;
    const int64_t s03 = a_src->nb[3] / ts0;

    const size_t ts_mul = ggml_type_size(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    const int64_t mul_s01 = mul_src->nb[1] / ts_mul;
    const int64_t mul_s02 = mul_src->nb[2] / ts_mul;
    const int64_t mul_s03 = mul_src->nb[3] / ts_mul;

    rms_norm_pre_add_mul_out_cuda<nv_bfloat16>(a_d, b_d, h_d, dst_d,
        ne00, ne01, ne02, ne03, s01, s02, s03,
        mul_d, mul_s01, mul_s02, mul_s03,
        mul_src->ne[0], mul_src->ne[1], mul_src->ne[2], mul_src->ne[3],
        eps, stream);
}

void ggml_cuda_rms_norm_gate_mul_bf16out(ggml_backend_cuda_context & ctx,
                                         const ggml_tensor *         rms_norm_tensor,
                                         const ggml_tensor *         mul_tensor,
                                         const ggml_tensor *         silu_tensor,
                                         const ggml_tensor *         gate_mul_tensor,
                                         void *                      dst_bf16) {
    GGML_ASSERT(mul_tensor->src[0] == rms_norm_tensor || mul_tensor->src[1] == rms_norm_tensor);
    GGML_ASSERT(gate_mul_tensor->src[0] == silu_tensor || gate_mul_tensor->src[1] == silu_tensor);

    const ggml_tensor * x_src    = rms_norm_tensor->src[0];
    const ggml_tensor * w_src    = (mul_tensor->src[0] == rms_norm_tensor) ? mul_tensor->src[1] : mul_tensor->src[0];
    const ggml_tensor * gate_src = silu_tensor->src[0];

    float eps = 0.0f;
    memcpy(&eps, rms_norm_tensor->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    const float * x_d    = (const float *) x_src->data;
    const float * w_d    = (const float *) w_src->data;
    const float * gate_d = (const float *) gate_src->data;
    nv_bfloat16 * dst_d  = (nv_bfloat16 *)  dst_bf16;
    cudaStream_t  stream = ctx.stream();

    GGML_ASSERT(x_src->type    == GGML_TYPE_F32);
    GGML_ASSERT(w_src->type    == GGML_TYPE_F32);
    GGML_ASSERT(gate_src->type == GGML_TYPE_F32);
    GGML_ASSERT(rms_norm_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(mul_tensor->type      == GGML_TYPE_F32);
    GGML_ASSERT(silu_tensor->type     == GGML_TYPE_F32);

    const int64_t ne00 = rms_norm_tensor->ne[0];
    const int64_t ne01 = rms_norm_tensor->ne[1];
    const int64_t ne02 = rms_norm_tensor->ne[2];
    const int64_t ne03 = rms_norm_tensor->ne[3];

    const size_t ts0 = ggml_type_size(x_src->type);
    GGML_ASSERT(x_src->nb[0] == ts0);
    const int64_t s01 = x_src->nb[1] / ts0;
    const int64_t s02 = x_src->nb[2] / ts0;
    const int64_t s03 = x_src->nb[3] / ts0;

    const size_t ts_mul = ggml_type_size(w_src->type);
    GGML_ASSERT(w_src->nb[0] == ts_mul);
    const int64_t mul_s01 = w_src->nb[1] / ts_mul;
    const int64_t mul_s02 = w_src->nb[2] / ts_mul;
    const int64_t mul_s03 = w_src->nb[3] / ts_mul;

    const size_t ts_gate = ggml_type_size(gate_src->type);
    GGML_ASSERT(gate_src->nb[0] == ts_gate);
    const int64_t gate_s01 = gate_src->nb[1] / ts_gate;
    const int64_t gate_s02 = gate_src->nb[2] / ts_gate;
    const int64_t gate_s03 = gate_src->nb[3] / ts_gate;

    rms_norm_gate_mul_out_cuda<nv_bfloat16>(x_d, dst_d,
        ne00, ne01, ne02, ne03, s01, s02, s03,
        w_d, mul_s01, mul_s02, mul_s03,
        w_src->ne[0], w_src->ne[1], w_src->ne[2], w_src->ne[3],
        gate_d, gate_s01, gate_s02, gate_s03,
        gate_src->ne[0], gate_src->ne[1], gate_src->ne[2], gate_src->ne[3],
        eps, stream);
}
