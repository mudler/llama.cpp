#pragma once
// On-demand paged KV block allocation (patch 0004, experimental).
//
// Backs the paged placement in llama_kv_cache::find_slot (patch 0002) with the
// vendored host-side PagedKVManager (patch 0001). Instead of mapping a
// sequence's logical positions onto a fixed full-pool permutation, blocks are
// popped from a free pool ON DEMAND as the sequence crosses block boundaries,
// and returned to the pool on sequence end. This is where the paged memory-
// capacity benefit begins: a short sequence holds only a few blocks, not the
// whole reserved window.
//
// Gated behind env LLAMA_KV_PAGED; a no-op when unset. All state lives in this
// unit (a static registry keyed by kv-cache + stream), so the core kv-cache
// struct stays untouched - find_slot only gains a gated call.

#include <cstdint>
#include <vector>

namespace paged_alloc {

// true iff env LLAMA_KV_PAGED is set (evaluated once).
bool active();

// Place n_tokens logical positions [base, base+n_tokens) of one stream on
// demand, appending their physical cell indices to `out`. pool_blocks =
// cells.size()/block_size is this stream's block budget. Returns false (leaving
// `out` unchanged) on pool exhaustion, so the caller falls back to the stock
// allocator. The caller still validates each returned cell is empty.
bool place(const void * cache, int stream, uint32_t base, uint32_t n_tokens,
           uint32_t block_size, uint32_t pool_blocks,
           std::vector<uint32_t> & out);

// Return a stream's blocks to the pool (sequence end).
void release(const void * cache, int stream);

// Return every stream's blocks for a kv-cache (clear() / teardown).
void release_all(const void * cache);

} // namespace paged_alloc
