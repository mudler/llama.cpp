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

// One persistent PagedKVManager per (kv-cache, stream): each stream owns a
// separate physical pool of cells.size() cells, so a manager's block ids map
// directly to cell ranges within that stream's pool. Requests inside a manager
// are keyed by the real llama_seq_id (NOT a fixed 0), so free(seq) releases one
// sequence and shared blocks survive at ref>0 - this is what makes ref-counted
// cross-request prefix sharing (0007) possible. Caching is enabled so commit()
// can publish blocks and share_prefix() can hit them.
std::map<key_t, std::unique_ptr<paged::PagedKVManager>> g_managers;

paged::PagedKVManager * get_mgr(const void * cache, int stream,
                                uint32_t pool_blocks, uint32_t block_size) {
    const key_t k{cache, stream};
    auto it = g_managers.find(k);
    if (it == g_managers.end()) {
        auto mgr = std::make_unique<paged::PagedKVManager>(
            (int32_t) pool_blocks, (int) block_size, /*enable_caching=*/true);
        it = g_managers.emplace(k, std::move(mgr)).first;
    }
    return it->second.get();
}

paged::PagedKVManager * find_mgr(const void * cache, int stream) {
    auto it = g_managers.find({cache, stream});
    return it == g_managers.end() ? nullptr : it->second.get();
}

} // namespace

bool place(const void * cache, int stream, int seq, uint32_t base, uint32_t n_tokens,
           uint32_t block_size, uint32_t pool_blocks,
           std::vector<uint32_t> & out) {
    if (n_tokens == 0) {
        return true;
    }

    paged::PagedKVManager * mgr = get_mgr(cache, stream, pool_blocks, block_size);

    const size_t before = mgr->block_table(seq).size();

    // Grow this sequence's request to cover its highest logical position. The
    // manager pops free blocks only for boundaries actually crossed; if
    // share_prefix() already reserved these blocks, this is a no-op.
    if (!mgr->allocate(seq, (size_t) base + n_tokens)) {
        return false; // pool exhausted -> caller falls back to the stock path
    }

    out.reserve(out.size() + n_tokens);
    for (uint32_t i = 0; i < n_tokens; ++i) {
        const int64_t s = mgr->slot(seq, (int) (base + i));
        out.push_back((uint32_t) s);
    }

    if (debug()) {
        const size_t after = mgr->block_table(seq).size();
        if (after != before) {
            fprintf(stderr,
                    "[paged-alloc] cache=%p stream=%d seq=%d grew %zu->%zu blocks "
                    "(budget=%u; base=%u +%u tok)\n",
                    cache, stream, seq, before, after, pool_blocks, base, n_tokens);
        }
    }

    return true;
}

size_t share_prefix(const void * cache, int stream, int seq,
                    const std::vector<int> & tokens,
                    uint32_t block_size, uint32_t pool_blocks) {
    paged::PagedKVManager * mgr = get_mgr(cache, stream, pool_blocks, block_size);
    const size_t shared_blocks = mgr->place_with_prefix(seq, tokens);
    const size_t shared_tokens = shared_blocks * (size_t) block_size;
    if (debug() && shared_blocks > 0) {
        fprintf(stderr,
                "[paged-alloc] cache=%p stream=%d seq=%d shares %zu prefix blocks "
                "(%zu tokens) - prefix NOT recomputed\n",
                cache, stream, seq, shared_blocks, shared_tokens);
    }
    return shared_tokens;
}

int64_t slot(const void * cache, int stream, int seq, int pos) {
    paged::PagedKVManager * mgr = find_mgr(cache, stream);
    if (!mgr) {
        return -1;
    }
    if ((size_t) (pos / mgr->block_size()) >= mgr->num_blocks(seq)) {
        return -1;
    }
    return mgr->slot(seq, pos);
}

void commit(const void * cache, int stream, int seq,
            const std::vector<int> & tokens, uint32_t block_size, uint32_t pool_blocks) {
    paged::PagedKVManager * mgr = get_mgr(cache, stream, pool_blocks, block_size);
    mgr->cache_blocks(seq, mgr->compute_block_hashes(tokens), tokens.size());
    if (debug()) {
        fprintf(stderr, "[paged-alloc] cache=%p stream=%d seq=%d committed %zu tokens\n",
                cache, stream, seq, tokens.size());
    }
}

void release(const void * cache, int stream, int seq) {
    paged::PagedKVManager * mgr = find_mgr(cache, stream);
    if (!mgr) {
        return;
    }
    mgr->free(seq); // ref-counted: shared blocks survive while another seq holds them
    if (debug()) {
        fprintf(stderr, "[paged-alloc] released cache=%p stream=%d seq=%d (free=%zu)\n",
                cache, stream, seq, mgr->num_free_blocks());
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

int ref_cnt_at(const void * cache, int stream, int seq, int pos, uint32_t block_size) {
    paged::PagedKVManager * mgr = find_mgr(cache, stream);
    if (!mgr) {
        return -1;
    }
    const size_t bi = (size_t) pos / block_size;
    if (bi >= mgr->num_blocks(seq)) {
        return -1;
    }
    return mgr->block_ref_cnt_at(seq, bi);
}

size_t num_free(const void * cache, int stream) {
    paged::PagedKVManager * mgr = find_mgr(cache, stream);
    return mgr ? mgr->num_free_blocks() : 0;
}

} // namespace paged_alloc
