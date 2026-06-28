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
class  llm_graph_input_attn_kv;

namespace paged_attn {

// true iff env LLAMA_KV_PAGED is set (evaluated once).
bool active();

// [S1] true iff the paged decode-graph reuse (layer-A can_reuse on the paged
// inputs) is ENABLED. Default ON when active(); LLAMA_PAGED_NO_GRAPH_REUSE=1
// forces it off (A/B probe / safety hatch). When off the paged inputs keep the
// stock default can_reuse()==false, i.e. the pre-S1 behaviour (rebuild every
// step). Bit-exact either way - reuse only skips the host-side graph rebuild,
// set_inputs still re-runs every step.
bool decode_graph_reuse();

// Gather K, V and the kq_mask down to the current sequence's non-empty cells.
// No-op (returns immediately) unless active(). On return *k, *v and *kq_mask
// point at the compacted tensors; pass them straight to build_attn_mha.
// `owner` is the attention input that owns the live (per-decode-refreshed) memory
// context; the paged input reads owner->mctx in can_reuse so a reused graph picks
// up the fresh context (see input_gather_idxs::can_reuse). May be null (no reuse).
void gather(ggml_context * ctx0,
            llm_graph_result * res,
            const llama_kv_cache_context * mctx,
            const llm_graph_input_attn_kv * owner,
            ggml_tensor ** k,
            ggml_tensor ** v,
            ggml_tensor ** kq_mask);

// [paged inc1] In-kernel paged decode read. Instead of materializing the
// sequence's cells (gather()), present K and V as n_gather-length VIEWS of the
// full physical window and return the position-ordered physical-cell index list
// as a block table (src[5] of ggml_flash_attn_ext). The fattn kernel/op then
// reads K_base + block_table[j]*nb in-kernel, removing the get_rows of K and V
// (the bulk of the gather). On return (true): *k,*v point at the views, *kq_mask
// at the compacted mask, *block_table at the I32 [n_gather, n_stream] index.
// Returns false (leaving *k,*v,*kq_mask untouched) when the in-kernel path does
// not apply - env off, nothing placed, or a transposed V cache - so the caller
// keeps the dense gather()/contiguous read.
bool in_kernel_decode(ggml_context * ctx0,
                      llm_graph_result * res,
                      const llama_kv_cache_context * mctx,
                      const llm_graph_input_attn_kv * owner,
                      ggml_tensor ** k,
                      ggml_tensor ** v,
                      ggml_tensor ** kq_mask,
                      ggml_tensor ** block_table);

} // namespace paged_attn
