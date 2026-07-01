#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <sstream>
#include <string>

struct server_admission_trace_step {
    int32_t n_batch = 0;
    int32_t n_ubatch = 0;
    int32_t decode_tokens = 0;
    int32_t prompt_tokens = 0;
    int32_t waiting_prompt_slots = 0;
    int32_t started_prompt_slots = 0;
    int32_t continued_prompt_slots = 0;
    int32_t ttft_deferred_decode_slots = 0;
    int32_t prefill_budget_step = 0;
    int32_t prefill_cap_per_slot = 0;
    bool decode_only_step = false;
};

struct server_admission_trace_totals {
    int64_t steps = 0;
    int64_t decode_only_steps = 0;
    int64_t decode_tokens = 0;
    int64_t prompt_tokens = 0;
    int64_t waiting_prompt_slots = 0;
    int64_t started_prompt_slots = 0;
    int64_t continued_prompt_slots = 0;
    int64_t ttft_deferred_decode_slots = 0;
    int32_t max_waiting_prompt_slots = 0;
    int32_t last_n_batch = 0;
    int32_t last_n_ubatch = 0;
    int32_t last_prefill_budget_step = 0;
    int32_t last_prefill_cap_per_slot = 0;
    std::array<int64_t, 6> prompt_hist = {};
    std::array<int64_t, 6> decode_hist = {};
    std::array<int64_t, 6> waiting_hist = {};
};

static inline int server_admission_trace_prompt_bucket(const int32_t n) {
    if (n <= 0) {
        return 0;
    }
    if (n <= 64) {
        return 1;
    }
    if (n <= 128) {
        return 2;
    }
    if (n <= 256) {
        return 3;
    }
    if (n <= 512) {
        return 4;
    }
    return 5;
}

static inline int server_admission_trace_decode_bucket(const int32_t n) {
    if (n <= 0) {
        return 0;
    }
    if (n <= 63) {
        return 1;
    }
    if (n <= 127) {
        return 2;
    }
    if (n <= 255) {
        return 3;
    }
    if (n <= 511) {
        return 4;
    }
    return 5;
}

static inline int server_admission_trace_waiting_bucket(const int32_t n) {
    if (n <= 0) {
        return 0;
    }
    if (n <= 7) {
        return 1;
    }
    if (n <= 15) {
        return 2;
    }
    if (n <= 31) {
        return 3;
    }
    if (n <= 63) {
        return 4;
    }
    return 5;
}

static inline void server_admission_trace_format_hist(
        std::ostringstream & out,
        const std::array<int64_t, 6> & hist,
        const char * const labels[6]) {
    bool first = true;
    for (int i = 0; i < 6; ++i) {
        if (hist[i] == 0) {
            continue;
        }
        if (!first) {
            out << ",";
        }
        out << labels[i] << ":" << hist[i];
        first = false;
    }
}

static inline void server_admission_trace_accumulate(
        server_admission_trace_totals & totals,
        const server_admission_trace_step & step) {
    totals.steps++;
    totals.decode_only_steps += step.decode_only_step ? 1 : 0;
    totals.decode_tokens += step.decode_tokens;
    totals.prompt_tokens += step.prompt_tokens;
    totals.waiting_prompt_slots += step.waiting_prompt_slots;
    totals.started_prompt_slots += step.started_prompt_slots;
    totals.continued_prompt_slots += step.continued_prompt_slots;
    totals.ttft_deferred_decode_slots += step.ttft_deferred_decode_slots;
    totals.max_waiting_prompt_slots = std::max(totals.max_waiting_prompt_slots, step.waiting_prompt_slots);
    totals.last_n_batch = step.n_batch;
    totals.last_n_ubatch = step.n_ubatch;
    totals.last_prefill_budget_step = step.prefill_budget_step;
    totals.last_prefill_cap_per_slot = step.prefill_cap_per_slot;
    totals.prompt_hist[server_admission_trace_prompt_bucket(step.prompt_tokens)]++;
    totals.decode_hist[server_admission_trace_decode_bucket(step.decode_tokens)]++;
    totals.waiting_hist[server_admission_trace_waiting_bucket(step.waiting_prompt_slots)]++;
}

static inline std::string server_admission_trace_format(const server_admission_trace_totals & totals) {
    std::ostringstream out;
    out << "serving admission trace:"
        << " steps=" << totals.steps
        << " decode_only_steps=" << totals.decode_only_steps
        << " decode_tokens=" << totals.decode_tokens
        << " prompt_tokens=" << totals.prompt_tokens
        << " waiting_prompt_slots=" << totals.waiting_prompt_slots
        << " max_waiting_prompt_slots=" << totals.max_waiting_prompt_slots
        << " started_prompt_slots=" << totals.started_prompt_slots
        << " continued_prompt_slots=" << totals.continued_prompt_slots
        << " ttft_deferred_decode_slots=" << totals.ttft_deferred_decode_slots
        << " last_n_batch=" << totals.last_n_batch
        << " last_n_ubatch=" << totals.last_n_ubatch
        << " last_prefill_budget_step=" << totals.last_prefill_budget_step
        << " last_prefill_cap_per_slot=" << totals.last_prefill_cap_per_slot;
    static const char * const prompt_labels[6] = {"0", "1-64", "65-128", "129-256", "257-512", "513+"};
    static const char * const decode_labels[6] = {"0", "1-63", "64-127", "128-255", "256-511", "512+"};
    static const char * const waiting_labels[6] = {"0", "1-7", "8-15", "16-31", "32-63", "64+"};
    out << " prompt_hist=";
    server_admission_trace_format_hist(out, totals.prompt_hist, prompt_labels);
    out << " decode_hist=";
    server_admission_trace_format_hist(out, totals.decode_hist, decode_labels);
    out << " waiting_hist=";
    server_admission_trace_format_hist(out, totals.waiting_hist, waiting_labels);
    return out.str();
}
