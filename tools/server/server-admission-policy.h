#pragma once

#include <cstdint>

static inline bool server_admission_should_defer_decode_for_ttft(
        const bool enabled,
        const bool prompt_waiting,
        const int32_t n_decoded) {
    return enabled && prompt_waiting && n_decoded > 0;
}

static inline bool server_admission_should_defer_decode_for_ttft(
        const bool enabled,
        const bool prompt_waiting,
        const int32_t n_decoded,
        const int32_t deferred_decode_slots,
        const int32_t max_deferred_decode_slots) {
    return server_admission_should_defer_decode_for_ttft(enabled, prompt_waiting, n_decoded) &&
        (max_deferred_decode_slots <= 0 || deferred_decode_slots < max_deferred_decode_slots);
}
