#pragma once

#include <cstdint>

static inline bool server_admission_should_defer_decode_for_ttft(
        const bool enabled,
        const bool prompt_waiting,
        const int32_t n_decoded) {
    return enabled && prompt_waiting && n_decoded > 0;
}
