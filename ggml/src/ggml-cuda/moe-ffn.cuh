#pragma once

#include "common.cuh"

bool ggml_cuda_compute_forward(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_moe_routed_ffn_poc_enabled();

bool ggml_cuda_moe_routed_ffn_poc_should_engage(
    const ggml_tensor * gate_up,
    const ggml_tensor * gate,
    const ggml_tensor * up,
    const ggml_tensor * glu,
    const ggml_tensor * down,
    const ggml_tensor * ids,
    int cc);

bool ggml_cuda_moe_routed_ffn_poc(
    ggml_backend_cuda_context & ctx,
    ggml_tensor * gate_up,
    ggml_tensor * gate,
    ggml_tensor * up,
    ggml_tensor * glu,
    ggml_tensor * down);
