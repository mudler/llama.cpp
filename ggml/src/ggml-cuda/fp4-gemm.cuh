#pragma once

#include "common.cuh"

// [paged patch 0034] Native NVFP4 (W4A4) large-M GEMM for Blackwell sm_121a (GB10).
//
// A Marlin-class tiled FP4-MMA GEMM (cp.async multistage prefetch, register-resident
// accumulators, ldmatrix A-operand, m16n8k64 mxf4nvf4 block-scale OMMA with e4m3
// true-scale) that beats the dequant->bf16 cuBLAS (nvjet) path that the rejected 0033
// scaffold routed large-M prefill through. The kernel body is the bit-exact PoC
// (NMSE=0 vs a same-dequant f32 reference) at its tuned best config
// (128x128 / KBLK4 / STAGES2 / PAD4).
//
// It is bit-exact-by-construction with the shipped FP4-MMQ path: it consumes the SAME
// e2m1 weight nibbles + e4m3 scale bytes from the GGUF block_nvfp4, quantizes
// activations with the SAME math as quantize_mmq_nvfp4 (e4m3 amax/6 scale + the +/-2
// code search + ggml_cuda_float_to_fp4_e2m1), and feeds the SAME hardware OMMA. The
// only difference vs FP4-MMQ is the K-accumulation order (a different but equivalent
// f32 reduction tree), which is greedy-md5 gated like every other paged path.
//
// Engages ONLY at large M (prefill), behind the 0033 LLAMA_FP4_PREFILL_M threshold;
// decode and small-M are byte-untouched and never reach this kernel.

// True if the native FP4 large-M path should handle this dense NVFP4 mul_mat:
//   src0 NVFP4 + src1/dst f32, contiguous, not transposed, 2D, Blackwell,
//   LLAMA_FP4_PREFILL_M > 0, M = src1->ne[1] > threshold, N % 128 == 0, K % 256 == 0.
// This single predicate also routes per-expert MoE slices (they flow through
// ggml_cuda_mul_mat) into the native kernel.
bool ggml_cuda_fp4_prefill_should_engage(
        const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc);

// Native FP4 W4A4 GEMM: dst[M,N] = src1_act[M,K] @ src0_w[N,K]^T.
// src0 = NVFP4 weights, src1 = f32 activations, dst = f32. Streams on ctx.stream(),
// pool-allocates scratch; no host sync. Caller must have checked
// ggml_cuda_fp4_prefill_should_engage().
void ggml_cuda_mul_mat_fp4_large_m(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
