#pragma once
// Paged attention gather-read (patch 0003, experimental).
//
// Companion to the paged block placement in llama_kv_cache::find_slot (patch
// 0002). Patch 0002 places a sequence's tokens at permuted, non-contiguous
// fixed-size block cells, but attention still reads the whole [0, n_kv) window
// (empty cells masked to -inf). This unit compacts that read: it gathers K, V
// and the kq_mask down to ONLY the sequence's used (non-empty) cells before
// build_attn_mha.
//
// Correctness: attention is permutation-invariant over the KV set, and dropping
// already-masked empty cells removes only exp(-inf)=0 terms - so greedy output
// is identical to stock. Gated behind env LLAMA_KV_PAGED; a no-op when unset.
//
// All logic lives here to keep the core files additive: build_attn gets one
// call, llama_kv_cache_context gets two thin accessors, CMake gets one line.

#include <cstddef>
#include <cstdint>

struct ggml_context;
struct ggml_tensor;
class  llm_graph_result;
class  llama_kv_cache_context;

namespace paged_attn {

// true iff env LLAMA_KV_PAGED is set (evaluated once).
bool active();

// Gather K, V and the kq_mask down to the current sequence's non-empty cells.
// No-op (returns immediately) unless active(). On return *k, *v and *kq_mask
// point at the compacted tensors; pass them straight to build_attn_mha.
void gather(ggml_context * ctx0,
            llm_graph_result * res,
            const llama_kv_cache_context * mctx,
            ggml_tensor ** k,
            ggml_tensor ** v,
            ggml_tensor ** kq_mask);

} // namespace paged_attn
