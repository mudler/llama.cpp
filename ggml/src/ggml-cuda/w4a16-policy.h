#pragma once

#include "ggml.h"

#include <cstdint>

static inline bool ggml_cuda_w4a16_direct_a_should_engage_params(
        ggml_type src0_type,
        ggml_type src1_type,
        ggml_type dst_type,
        bool blackwell,
        int64_t w4a16_prefill_m,
        bool direct_a,
        int64_t tokens,
        int64_t k,
        int64_t n) {
    if (!direct_a || w4a16_prefill_m <= 0) {
        return false;
    }
    if (src0_type != GGML_TYPE_NVFP4 || src1_type != GGML_TYPE_F32 || dst_type != GGML_TYPE_F32) {
        return false;
    }
    if (!blackwell || tokens <= w4a16_prefill_m) {
        return false;
    }
    return k % 64 == 0 && n % 128 == 0;
}
