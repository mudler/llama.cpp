#include "paged-attn.h"

#include "llama-graph.h"
#include "llama-kv-cache.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdlib>
#include <cstdio>

namespace paged_attn {

bool active() {
    static const bool a = (std::getenv("LLAMA_KV_PAGED") != nullptr);
    return a;
}

static bool debug() {
    static const bool d = (std::getenv("LLAMA_KV_PAGED_DEBUG") != nullptr);
    return d;
}

namespace {

// Graph input that, at set_input time, fills an I32 [n_gather, n_stream] tensor
// with each stream's non-empty cell indices (position-sorted, padded with a
// masked/empty cell) by delegating to the kv-cache context. Private to this
// unit; default can_reuse()==false keeps the graph from being reused across
// decodes (n_gather grows every step).
class input_gather_idxs : public llm_graph_input_i {
public:
    input_gather_idxs(const llama_kv_cache_context * mctx, ggml_tensor * idxs)
        : mctx(mctx), idxs(idxs) {}

    void set_input(const llama_ubatch * ubatch) override {
        GGML_UNUSED(ubatch);
        GGML_ASSERT(idxs && ggml_backend_buffer_is_host(idxs->buffer));
        mctx->get_gather_idxs((int32_t *) idxs->data);
    }

    const llama_kv_cache_context * mctx;
    ggml_tensor * idxs;
};

} // namespace

void gather(ggml_context * ctx0,
            llm_graph_result * res,
            const llama_kv_cache_context * mctx,
            ggml_tensor ** k,
            ggml_tensor ** v,
            ggml_tensor ** kq_mask) {
    if (!active()) {
        return;
    }

    ggml_tensor * K = *k;
    ggml_tensor * V = *v;
    ggml_tensor * M = *kq_mask;

    // Number of streams (sequences) in the unified batch. K is laid out
    // [d, h, n_kv, n_stream] and the mask is [n_kv, n_tps, 1, n_stream]; the
    // gather is per-stream (one index column per stream), so a single
    // ggml_get_rows over the stream axis handles 1..N streams uniformly.
    const int64_t n_stream = K->ne[3];
    GGML_ASSERT(M->ne[3] == n_stream);

    const int64_t n_gather = (int64_t) mctx->get_n_gather();
    if (n_gather <= 0) {
        // Worst-case graph reserve (empty cache) or nothing placed yet: leave
        // the full [0, n_kv) read untouched so buffer sizing stays worst-case.
        return;
    }

    if (debug()) {
        static int64_t once = 0;
        if (once++ < 2) {
            fprintf(stderr, "[paged-attn] gather n_stream=%lld n_kv=%lld n_gather=%lld\n",
                    (long long) n_stream, (long long) K->ne[2], (long long) n_gather);
        }
    }

    // Per-stream index tensor [n_gather, n_stream], filled at set_input from
    // each stream's non-empty cells. ggml_get_rows broadcasts along ne[1]==
    // n_stream, so column s gathers from stream s of the source.
    ggml_tensor * idx = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_gather, n_stream);
    ggml_set_input(idx);
    res->add_input(llm_graph_input_ptr(new input_gather_idxs(mctx, idx)));

    // --- gather K: collapse (head_dim, n_head) so cells become the row axis ---
    {
        ggml_tensor * t = ggml_cont(ctx0, K);                                          // [d, h, n_kv, ns]
        t = ggml_reshape_3d(ctx0, t, K->ne[0]*K->ne[1], K->ne[2], n_stream);           // [d*h, n_kv, ns]
        t = ggml_get_rows(ctx0, t, idx);                                               // [d*h, n_gather, ns]
        *k = ggml_reshape_4d(ctx0, t, K->ne[0], K->ne[1], n_gather, n_stream);         // [d, h, n_gather, ns]
    }

    // --- gather V ---
    // Normalize to a non-transposed [d, h, n_kv, ns] view first, so the gathered
    // result is contiguous and build_attn_mha sees a consistent v_trans==false.
    {
        const bool v_trans = V->nb[1] > V->nb[2];
        ggml_tensor * vsrc = v_trans
            ? ggml_permute(ctx0, V, 2, 1, 0, 3)   // [n_kv, h, d, ns] -> [d, h, n_kv, ns]
            : V;                                  // already [d, h, n_kv, ns]
        ggml_tensor * t = ggml_cont(ctx0, vsrc);                                       // [d, h, n_kv, ns]
        t = ggml_reshape_3d(ctx0, t, vsrc->ne[0]*vsrc->ne[1], vsrc->ne[2], n_stream);  // [d*h, n_kv, ns]
        t = ggml_get_rows(ctx0, t, idx);                                               // [d*h, n_gather, ns]
        *v = ggml_reshape_4d(ctx0, t, vsrc->ne[0], vsrc->ne[1], n_gather, n_stream);   // [d, h, n_gather, ns]
    }

    // --- gather mask (cells are ne0): transpose so cells become the row axis,
    //     gather per stream, transpose back ---
    {
        ggml_tensor * m = ggml_reshape_3d(ctx0, M, M->ne[0], M->ne[1], n_stream);      // [n_kv, n_tps, ns]
        m = ggml_cont(ctx0, ggml_transpose(ctx0, m));                                  // [n_tps, n_kv, ns]
        m = ggml_get_rows(ctx0, m, idx);                                               // [n_tps, n_gather, ns] (F32)
        m = ggml_cont(ctx0, ggml_transpose(ctx0, m));                                  // [n_gather, n_tps, ns]
        m = ggml_reshape_4d(ctx0, m, n_gather, M->ne[1], 1, n_stream);
        if (M->type != m->type) {
            m = ggml_cast(ctx0, m, M->type);   // flash-attn requires an F16 mask
        }
        *kq_mask = m;
    }
}

} // namespace paged_attn
