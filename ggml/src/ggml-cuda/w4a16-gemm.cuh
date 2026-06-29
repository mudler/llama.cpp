#pragma once

#include "common.cuh"

// [paged patch 0035] Marlin-style W4A16 GROUPED MoE prefill GEMM for Blackwell sm_121a (GB10).
//
// This is the profile-validated #2 prefill lever and a DISTINCT kernel from the two prefill
// rejects:
//   - NOT patch 0033 (separate-pass dequant -> bf16 cuBLAS / nvjet): that pays a full per-step
//     weight dequant + 4x bf16 weight traffic and lost to FP4-MMQ at large M.
//   - NOT patch 0034 (native W4A4 FP4-MMA, mxf4nvf4 block-scale OMMA): that quantizes the
//     activations to FP4 and so still pays the quantize_mmq_nvfp4 activation-quant tax.
//
// The winning shape vLLM uses on this silicon (Marlin W4A16): the FP4 expert weights are
// dequantized to bf16 IN REGISTERS right before the MMA (never materialized to global/smem as
// bf16), the activations stay bf16 (a cheap f32->bf16 cast, NO per-block FP4 amax/code-search
// quantize), and the product is a standard bf16 m16n8k16 mma.sync feeding f32 accumulators,
// cp.async multistage-pipelined over the K loop. So W4A16 pays ZERO activation-quant (the paged
// FP4-MMQ path's quantize_mmq_nvfp4 is +15 us/tok) and the GEMM runs as a bf16 tensor-core GEMM
// with the weight read at 4 bits.
//
// GROUPED: the kernel is launched ONCE over the whole mul_mat_id token-sorted activation buffer
// (src1_sorted is already sorted-by-expert by the existing host-loop), with a per-M-tile expert
// map so each output tile reads its expert's weight matrix (src0 + expert*nb02) and the ragged
// per-expert row tail is masked. No per-expert kernel launch, no per-expert M-padding waste.
//
// Engages ONLY at large aggregate-M (prefill), behind LLAMA_W4A16_PREFILL_M (default 0 == OFF
// == stock); decode (small ne12) and the non-MoE / non-NVFP4 paths are byte-untouched. The bf16
// tiles are mma.cuh's (mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32).

// True if the grouped W4A16 path should handle this mul_mat_id:
//   src0 NVFP4, src1 f32, dst f32, Blackwell, LLAMA_W4A16_PREFILL_M>0,
//   ne12 (total tokens / aggregate prefill M) > threshold, N=ne0 % 128 == 0, K=ne10 % 64 == 0.
bool ggml_cuda_w4a16_moe_grouped_should_engage(
        const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc);

// True iff LLAMA_W4A16_PREFILL_M > 0 (the master on/off for the mmq.cu grouped-MMQ-off gate).
bool ggml_cuda_w4a16_prefill_enabled();
int64_t ggml_cuda_w4a16_prefill_m();

// Grouped W4A16 MoE GEMM over the token-sorted buffer.
//   src0          : NVFP4 weights [K, N, n_experts] (one [K,N] matrix per expert)
//   src1_sorted   : f32 [K, total_rows], rows already sorted by expert (the mul_mat_id host-loop's
//                   src1_sorted), with tokens_per_expert[e] consecutive rows per expert e
//   dst_sorted    : f32 [N, total_rows], written in the same sorted order
//   tokens_per_expert : host vector, length n_experts
// Streams on `stream`, pool-allocates scratch (bf16 activations + device tile map); no host sync.
void ggml_cuda_mul_mat_id_w4a16_grouped(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const float * src1_sorted,
        float * dst_sorted,
        const int * tokens_per_expert,
        int64_t n_experts, int64_t K, int64_t N,
        cudaStream_t stream);
