#include "common.cuh"

// [P1 bf16-stream] bf16-resident execution-pass helpers (default-off, gated by
// LLAMA_BF16_STREAM at the ggml_cuda_try_fuse call site). Siblings of the plain
// rms_norm and the 0042/0044 fused norms in norm.cu; these variants write a bf16
// output so the consuming projection GEMM reads the activation directly (no f32->bf16
// convert_dtype glue). Each kernel is templated on the output dtype (float or
// nv_bfloat16) so the file carries the full op-variant set the P1 contract names;
// the live q36 engage path instantiates the bf16 output only.
//
// All three keep the same reduction and FP order as their f32 originals; only the
// final store is narrowed to bf16 via __float2bfloat16, so the opt-in path stays a
// pure dtype (KL-benign) change, not an algorithmic one.

// plain rms_norm + weight multiply -> bf16  (attention input norm, GDN input norm)
void ggml_cuda_rms_norm_mul_bf16out(ggml_backend_cuda_context & ctx,
                                    const ggml_tensor *         rms_norm_tensor,
                                    const ggml_tensor *         mul_tensor,
                                    void *                      dst_bf16);

// 0042 residual-add + rms_norm + weight multiply -> f32 residual (h_out) + bf16 normed
// (op-set completeness; on q36 the ffn/moe-input norm feeds MMQ experts so the engage
// path bails, but the kernel + entry keep the op-variant set whole and the sentinel
// exercises it).
void ggml_cuda_rms_norm_pre_add_mul_bf16out(ggml_backend_cuda_context & ctx,
                                            const ggml_tensor *         add_tensor,
                                            const ggml_tensor *         rms_norm_tensor,
                                            const ggml_tensor *         mul_tensor,
                                            void *                      dst_bf16);

// 0044 gated-DeltaNet output norm  scale*x*w*silu(z)  -> bf16  (ssm_out; the P0 segment)
void ggml_cuda_rms_norm_gate_mul_bf16out(ggml_backend_cuda_context & ctx,
                                         const ggml_tensor *         rms_norm_tensor,
                                         const ggml_tensor *         mul_tensor,
                                         const ggml_tensor *         silu_tensor,
                                         const ggml_tensor *         gate_mul_tensor,
                                         void *                      dst_bf16);
