#include "paged-alloc.h"
#include "paged-kv-manager.h"

#include <cstdlib>
#include <cstdio>
#include <map>
#include <memory>
#include <utility>

namespace paged_alloc {

bool active() {
    static const bool a = (std::getenv("LLAMA_KV_PAGED") != nullptr);
    return a;
}

static bool debug() {
    static const bool d = (std::getenv("LLAMA_KV_PAGED_DEBUG") != nullptr);
    return d;
}

namespace {

using key_t = std::pair<const void *, int>;

// One PagedKVManager per (kv-cache, stream): each stream owns a separate
// physical pool of cells.size() cells, so a manager's block ids map directly to
// cell ranges within that stream's pool. The internal request id is always 0.
std::map<key_t, std::unique_ptr<paged::PagedKVManager>> g_managers;

paged::PagedKVManager * get_mgr(const void * cache, int stream,
                                uint32_t pool_blocks, uint32_t block_size) {
    const key_t k{cache, stream};
    auto it = g_managers.find(k);
    if (it == g_managers.end()) {
        // enable_caching=false: prefix caching is a later patch; 0004 exercises
        // only on-demand allocate / free.
        auto mgr = std::make_unique<paged::PagedKVManager>(
            (int32_t) pool_blocks, (int) block_size, /*enable_caching=*/false);
        it = g_managers.emplace(k, std::move(mgr)).first;
    }
    return it->second.get();
}

} // namespace

bool place(const void * cache, int stream, uint32_t base, uint32_t n_tokens,
           uint32_t block_size, uint32_t pool_blocks,
           std::vector<uint32_t> & out) {
    if (n_tokens == 0) {
        return true;
    }

    paged::PagedKVManager * mgr = get_mgr(cache, stream, pool_blocks, block_size);

    const size_t before = mgr->block_table(0).size();

    // Grow the request to cover the highest logical position. The manager pops
    // free blocks only for the boundaries actually crossed - that is the on-
    // demand behavior; an already-covered range adds nothing.
    if (!mgr->allocate(0, (size_t) base + n_tokens)) {
        return false; // pool exhausted -> caller falls back to the stock path
    }

    out.reserve(out.size() + n_tokens);
    for (uint32_t i = 0; i < n_tokens; ++i) {
        const int64_t s = mgr->slot(0, (int) (base + i));
        out.push_back((uint32_t) s);
    }

    if (debug()) {
        const size_t after = mgr->block_table(0).size();
        if (after != before) {
            fprintf(stderr,
                    "[paged-alloc] cache=%p stream=%d grew %zu->%zu blocks "
                    "(budget=%u; base=%u +%u tok)\n",
                    cache, stream, before, after, pool_blocks, base, n_tokens);
        }
    }

    return true;
}

void release(const void * cache, int stream) {
    auto it = g_managers.find({cache, stream});
    if (it == g_managers.end()) {
        return;
    }
    it->second->free(0);
    g_managers.erase(it);
    if (debug()) {
        fprintf(stderr, "[paged-alloc] released cache=%p stream=%d\n", cache, stream);
    }
}

void release_all(const void * cache) {
    for (auto it = g_managers.begin(); it != g_managers.end(); ) {
        if (it->first.first == cache) {
            it = g_managers.erase(it);
        } else {
            ++it;
        }
    }
}

} // namespace paged_alloc
