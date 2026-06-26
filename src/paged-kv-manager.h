#pragma once
// Paged KV cache block manager for llama.cpp (CPU-first prototype).
//
// Host-side block management is a faithful port of vLLM V1:
//   vllm/v1/core/kv_cache_utils.py            (KVCacheBlock, FreeKVCacheBlockQueue, hash_block_tokens)
//   vllm/v1/core/block_pool.py                (BlockPool: get_new_blocks/touch/free/evict/cache_full_blocks)
//   vllm/v1/core/single_type_kv_cache_manager.py (allocate_new_blocks, find_longest_cache_hit)
//
// Parity is on behavior/algorithm (block chaining, first-miss stop, ref-counting,
// LRU eviction order), not on exact hash bytes. This unit has zero ggml/llama.cpp
// dependency so it can be unit-tested in isolation.

#include <cstddef>
#include <cstdint>
#include <vector>
#include <unordered_map>
#include <map>
#include <utility>

namespace paged {

// vLLM KVCacheBlock (kv_cache_utils.py).
struct KVCacheBlock {
    int32_t  block_id   = 0;
    int      ref_cnt    = 0;
    bool     has_hash   = false;   // vLLM: _block_hash is set only when full+cached
    uint64_t block_hash = 0;
    bool     is_null    = false;
    KVCacheBlock* prev_free = nullptr;
    KVCacheBlock* next_free = nullptr;

    explicit KVCacheBlock(int32_t id = 0) : block_id(id) {}
    void reset_hash() { has_hash = false; block_hash = 0; }
};

// Intrusive doubly-linked free list with fake head/tail (vLLM FreeKVCacheBlockQueue).
// O(1) middle removal is required so touch() can pull a warm cached block out of the
// free list when a later request hits its prefix.
class FreeBlockQueue {
public:
    size_t num_free_blocks = 0;

    explicit FreeBlockQueue(const std::vector<KVCacheBlock*>& blocks);
    KVCacheBlock* popleft();
    std::vector<KVCacheBlock*> popleft_n(size_t n);
    void remove(KVCacheBlock* block);
    void append(KVCacheBlock* block);
    void append_n(const std::vector<KVCacheBlock*>& blocks);
    void prepend_n(const std::vector<KVCacheBlock*>& blocks);
    std::vector<KVCacheBlock*> get_all_free_blocks() const;
    // [paged 0024 Fix-2] Relink the intrusive free list to the given order using
    // THIS queue's fake head/tail (the nodes' addresses are stable; a temporary
    // FreeBlockQueue would leave dangling fake-node pointers). Used to restore a
    // pristine, contiguous popleft order after a fragmenting burst drains.
    void rebuild(const std::vector<KVCacheBlock*>& blocks);

private:
    KVCacheBlock fake_head{-1};
    KVCacheBlock fake_tail{-1};
};

// vLLM BlockPool (block_pool.py).
class BlockPool {
public:
    KVCacheBlock* null_block = nullptr;

    BlockPool(int32_t num_blocks, bool enable_caching);
    std::vector<KVCacheBlock*> get_new_blocks(size_t n);
    KVCacheBlock* get_cached_block(uint64_t block_hash);
    void touch(const std::vector<KVCacheBlock*>& blocks);
    void free_blocks(const std::vector<KVCacheBlock*>& ordered_blocks);
    void cache_full_blocks(const std::vector<KVCacheBlock*>& req_blocks,
                           size_t num_cached_blocks, size_t num_full_blocks,
                           const std::vector<uint64_t>& block_hashes);
    size_t get_num_free_blocks() const { return free_queue_.num_free_blocks; }
    // [paged 0024 Fix-2] Total non-null blocks, and whether the pool is fully
    // idle (every non-null block back in the free queue). defrag_free_queue()
    // relinks the free queue into pristine ascending-block-id order; only valid
    // when all_free() so no live request's block table is disturbed. Block hashes
    // are preserved, so a warm committed prefix stays re-hittable.
    size_t total_blocks() const { return blocks_.size(); }
    bool   all_free()    const { return free_queue_.num_free_blocks + 1 == blocks_.size(); }
    void   defrag_free_queue();

private:
    bool maybe_evict_cached_block(KVCacheBlock* block);

    bool enable_caching_;
    std::vector<KVCacheBlock> blocks_;     // owns all block descriptors
    std::vector<KVCacheBlock*> ptrs_;
    FreeBlockQueue free_queue_;
    // vLLM stores hash -> {block_id: block} to allow duplicate-content blocks; the
    // prototype keeps the last writer (single KV-cache group is sufficient for the wins).
    std::unordered_map<uint64_t, KVCacheBlock*> cached_block_hash_to_block_;
};

// Allocation + prefix-caching surface, ported from SingleTypeKVCacheManager /
// FullAttentionManager. Single KV-cache group; no extra_keys / eagle / spec-decode.
class PagedKVManager {
public:
    PagedKVManager(int32_t num_blocks, int block_size, bool enable_caching);

    // Grow seq_id to cover total_tokens slots. Returns false on OOM (free queue empty).
    bool allocate(int seq_id, size_t total_tokens);
    std::vector<int32_t> block_table(int seq_id) const;
    int64_t slot(int seq_id, int pos) const;
    std::vector<int64_t> slot_mapping(int seq_id, const std::vector<int>& positions) const;
    void free(int seq_id);
    int block_size() const { return block_size_; }

    // [paged 0024 Fix-1] Reclaim the trailing blocks of seq_id beyond logical
    // position n_keep: free every block at index >= ceil(n_keep/bs) (ref-counted,
    // mirroring vLLM's free of a truncated block suffix). Called on a partial tail
    // seq_rm [n_keep, end) so the manager's block accounting tracks the kv-cache
    // exactly instead of stranding the blocks whose cells were just cleared.
    void truncate(int seq_id, size_t n_keep);

    // [paged 0024 Fix-2] When no live request holds a block, relink the free
    // queue into pristine contiguous order (undo a burst's scrambled free order).
    void defrag_free_pool() { if (pool_.all_free()) pool_.defrag_free_queue(); }

    // Prefix caching (win 3).
    static uint64_t hash_block(uint64_t parent_hash, const std::vector<int>& token_ids);
    std::vector<uint64_t> compute_block_hashes(const std::vector<int>& token_ids) const;
    size_t get_computed_blocks(const std::vector<uint64_t>& block_hashes); // returns num cached tokens
    void cache_blocks(int seq_id, const std::vector<uint64_t>& block_hashes, size_t num_tokens);

    // Cross-request prefix caching + copy-on-write (patch 0006).
    //
    // Splice the longest cached prefix of token_ids into seq_id (reuse the
    // shared physical blocks, ref_cnt++ so a block frees only at ref 0) and
    // allocate fresh blocks only for the divergent suffix. Returns the number of
    // shared (reused) blocks; the caller skips recomputing those tokens. On pool
    // exhaustion the sequence is rolled back (no ref leak) and 0 is returned.
    size_t place_with_prefix(int seq_id, const std::vector<int>& token_ids);

    // Copy-on-write the block at logical index bi of seq_id. If that block is
    // shared (ref_cnt>1), allocate a fresh private block, drop this seq's ref on
    // the shared one (other owners keep it, content untouched) and install the
    // fresh block at bi. Returns {old_block_id, new_block_id}; new==old when the
    // block was already private (ref_cnt<=1) and no copy is needed. The caller
    // copies the physical cell contents old_block_id -> new_block_id.
    std::pair<int32_t, int32_t> cow_block(int seq_id, size_t bi);

    // Introspection for the prefix-share gate (debug/tests).
    int    block_ref_cnt_at(int seq_id, size_t bi) const;
    size_t num_blocks(int seq_id) const;
    size_t num_free_blocks() const { return pool_.get_num_free_blocks(); }

protected:
    int block_size_;
    BlockPool pool_;
    std::map<int, std::vector<KVCacheBlock*>> req_to_blocks_;
};

} // namespace paged
