#pragma once
// On-demand paged KV block allocation + cross-request prefix reuse
// (patches 0004 + 0007, experimental).
//
// Backs the paged placement in llama_kv_cache::find_slot with the vendored
// host-side PagedKVManager (patch 0001). Two responsibilities:
//
//   * On-demand allocation (0004): a sequence's logical positions are mapped to
//     physical cells block-by-block, popped from a free pool only as the
//     sequence grows and returned on sequence end.
//
//   * Cross-request prefix reuse (0007): before a new sequence's suffix is
//     decoded, share_prefix() reuses the cached physical blocks of a matching
//     content prefix (ref_cnt++), so the engine shares the already-computed KV
//     cells and the caller decodes ONLY the divergent suffix - the prefix is not
//     recomputed. commit() publishes a sequence's full blocks into the content
//     cache so later sequences can hit them. Freeing is ref-counted: a shared
//     block returns to the pool only when every sharer has been released.
//
// One persistent PagedKVManager per (kv-cache, stream); requests inside it are
// keyed by the real llama_seq_id, so free(seq) releases exactly one sequence and
// shared blocks survive at ref>0. All state lives in this unit (a static
// registry), so the core kv-cache struct stays untouched - find_slot gains only
// gated calls. Gated behind env LLAMA_KV_PAGED; a no-op when unset.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace paged_alloc {

// true iff env LLAMA_KV_PAGED is set (evaluated once).
bool active();

// Place n_tokens logical positions [base, base+n_tokens) of (cache,stream,seq)
// on demand, appending their physical cell indices to `out`. pool_blocks =
// cells.size()/block_size is the stream's block budget. Returns false (leaving
// `out` unchanged) on pool exhaustion, so the caller falls back to the stock
// allocator. The caller still validates each returned cell is empty.
bool place(const void * cache, int stream, int seq, uint32_t base, uint32_t n_tokens,
           uint32_t block_size, uint32_t pool_blocks,
           std::vector<uint32_t> & out);

// [0007] Reuse the longest cached content prefix of `tokens` for (cache,stream,
// seq): splice the shared physical blocks into seq (ref_cnt++) and reserve fresh
// blocks for the divergent suffix. Returns the number of shared PREFIX TOKENS
// (block-aligned); the caller marks those cells for seq and decodes only the
// suffix. 0 if nothing matched or on pool exhaustion (sequence rolled back).
size_t share_prefix(const void * cache, int stream, int seq,
                    const std::vector<int> & tokens,
                    uint32_t block_size, uint32_t pool_blocks);

// [0007] Physical cell backing logical position `pos` of (cache,stream,seq), or
// -1 if seq is unknown. Used to map a shared prefix position to its cell.
int64_t slot(const void * cache, int stream, int seq, int pos);

// [0007] Publish seq's full (block-aligned) blocks into the content cache so a
// later share_prefix() can reuse them. Call after the sequence's KV is computed.
void commit(const void * cache, int stream, int seq,
            const std::vector<int> & tokens, uint32_t block_size, uint32_t pool_blocks);

// Return one sequence's blocks to the pool (ref-counted; sequence end).
void release(const void * cache, int stream, int seq);

// Drop every manager for a kv-cache (clear() / teardown).
void release_all(const void * cache);

// Introspection for the prefix-share gate (debug/tests). ref_cnt_at returns the
// ref count of the block backing logical position `pos`, or -1 if unknown.
int    ref_cnt_at(const void * cache, int stream, int seq, int pos, uint32_t block_size);
size_t num_free(const void * cache, int stream);

} // namespace paged_alloc
