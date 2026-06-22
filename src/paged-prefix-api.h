#pragma once
// Thin test/diagnostic shim over the paged cross-request prefix engine seam
// (patch 0007). Lets a driver that only includes the public llama.h reach the
// gated llama_kv_cache::paged_prefix_* methods and the paged-alloc introspection
// without pulling in the internal kv-cache headers. All entry points are no-ops
// (return 0) unless env LLAMA_KV_PAGED is set. Experimental; not a stable API.

#include <cstddef>
#include <cstdint>
#include "llama.h"

namespace paged_prefix_api {

// Reuse the longest cached content prefix of [tokens, tokens+n) for `seq` and
// return the number of shared prefix tokens (the caller decodes only the
// suffix). 0 if nothing was shared.
int32_t share(llama_context * ctx, llama_seq_id seq, const llama_token * tokens, int n);

// Publish `seq`'s full blocks into the content cache (call after its KV is computed).
void commit(llama_context * ctx, llama_seq_id seq, const llama_token * tokens, int n);

// Ref count of the paged block backing logical position `pos` of `seq` (unified
// stream 0), or -1 if unknown.
int ref_at(llama_context * ctx, llama_seq_id seq, int pos);

// Number of free blocks in the unified stream-0 pool, or 0 if no manager.
long num_free(llama_context * ctx);

} // namespace paged_prefix_api
