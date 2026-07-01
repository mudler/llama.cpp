#pragma once

#include <algorithm>
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
    int32_t max_waiting_prompt_slots = 0;
    int32_t last_n_batch = 0;
    int32_t last_n_ubatch = 0;
    int32_t last_prefill_budget_step = 0;
    int32_t last_prefill_cap_per_slot = 0;
};

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
    totals.max_waiting_prompt_slots = std::max(totals.max_waiting_prompt_slots, step.waiting_prompt_slots);
    totals.last_n_batch = step.n_batch;
    totals.last_n_ubatch = step.n_ubatch;
    totals.last_prefill_budget_step = step.prefill_budget_step;
    totals.last_prefill_cap_per_slot = step.prefill_cap_per_slot;
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
        << " last_n_batch=" << totals.last_n_batch
        << " last_n_ubatch=" << totals.last_n_ubatch
        << " last_prefill_budget_step=" << totals.last_prefill_budget_step
        << " last_prefill_cap_per_slot=" << totals.last_prefill_cap_per_slot;
    return out.str();
}
