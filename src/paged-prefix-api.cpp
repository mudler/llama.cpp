#include "paged-prefix-api.h"
#include "paged-alloc.h"
#include "llama-kv-cache.h"

#include <vector>

namespace paged_prefix_api {

static llama_kv_cache * kv_of(llama_context * ctx) {
    // The driver targets a plain unified KV-cache model; dynamic_cast yields null
    // for wrapped caches (iSWA / hybrid), where cross-request cell sharing does
    // not apply, so the shim degrades to a safe no-op.
    return dynamic_cast<llama_kv_cache *>(llama_get_memory(ctx));
}

int32_t share(llama_context * ctx, llama_seq_id seq, const llama_token * tokens, int n) {
    llama_kv_cache * kv = kv_of(ctx);
    if (!kv || n <= 0) {
        return 0;
    }
    return kv->paged_prefix_share(seq, std::vector<llama_token>(tokens, tokens + n));
}

void commit(llama_context * ctx, llama_seq_id seq, const llama_token * tokens, int n) {
    llama_kv_cache * kv = kv_of(ctx);
    if (!kv || n <= 0) {
        return;
    }
    kv->paged_prefix_commit(seq, std::vector<llama_token>(tokens, tokens + n));
}

int ref_at(llama_context * ctx, llama_seq_id seq, int pos) {
    llama_kv_cache * kv = kv_of(ctx);
    if (!kv) {
        return -1;
    }
    return paged_alloc::ref_cnt_at((const void *) kv, /*stream=*/0, (int) seq, pos, /*block_size=*/16);
}

long num_free(llama_context * ctx) {
    llama_kv_cache * kv = kv_of(ctx);
    if (!kv) {
        return 0;
    }
    return (long) paged_alloc::num_free((const void *) kv, /*stream=*/0);
}

} // namespace paged_prefix_api
