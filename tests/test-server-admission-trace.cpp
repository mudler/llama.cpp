#include "../tools/server/server-admission-trace.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

static void require(bool ok, const char * what) {
    if (!ok) {
        std::fprintf(stderr, "require failed: %s\n", what);
        std::exit(1);
    }
}

static void require_contains(const std::string & text, const char * needle) {
    if (text.find(needle) == std::string::npos) {
        std::fprintf(stderr, "missing '%s' in:\n%s\n", needle, text.c_str());
        std::exit(1);
    }
}

int main() {
    server_admission_trace_totals totals;

    server_admission_trace_step first;
    first.n_batch = 2048;
    first.n_ubatch = 512;
    first.decode_tokens = 128;
    first.prompt_tokens = 384;
    first.waiting_prompt_slots = 6;
    first.started_prompt_slots = 2;
    first.continued_prompt_slots = 1;
    first.decode_only_step = false;
    first.prefill_budget_step = 512;
    first.prefill_cap_per_slot = 512;
    server_admission_trace_accumulate(totals, first);

    server_admission_trace_step second;
    second.n_batch = 2048;
    second.n_ubatch = 512;
    second.decode_tokens = 128;
    second.prompt_tokens = 0;
    second.waiting_prompt_slots = 4;
    second.decode_only_step = true;
    server_admission_trace_accumulate(totals, second);

    const std::string summary = server_admission_trace_format(totals);

    require(totals.steps == 2, "step count");
    require(totals.decode_only_steps == 1, "decode-only count");
    require(totals.prompt_tokens == 384, "prompt token sum");
    require(totals.max_waiting_prompt_slots == 6, "max waiting prompt slots");

    require_contains(summary, "serving admission trace:");
    require_contains(summary, "steps=2");
    require_contains(summary, "decode_only_steps=1");
    require_contains(summary, "decode_tokens=256");
    require_contains(summary, "prompt_tokens=384");
    require_contains(summary, "max_waiting_prompt_slots=6");
    require_contains(summary, "started_prompt_slots=2");
    require_contains(summary, "continued_prompt_slots=1");
    require_contains(summary, "last_n_batch=2048");
    require_contains(summary, "last_n_ubatch=512");
    require_contains(summary, "last_prefill_budget_step=0");
    require_contains(summary, "last_prefill_cap_per_slot=0");

    return 0;
}
