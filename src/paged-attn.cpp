#include "paged-attn.h"

#include "llama-graph.h"
#include "llama-kv-cache.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdlib>
#include <cstdio>
#include <ctime>
namespace { static inline double l5_now_ns(){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return (double)ts.tv_sec*1e9+(double)ts.tv_nsec; } }
double g_l5_t_gbt=0, g_l5_t_setinp=0, g_l5_t_hostproc=0; long g_l5_n_gbt=0, g_l5_n_setinp=0, g_l5_n_hostproc=0;
extern "C" void l5_add_setinp(double ns){ g_l5_t_setinp+=ns; g_l5_n_setinp++; }
extern "C" void l5_add_hostproc(double ns){ g_l5_t_hostproc+=ns; g_l5_n_hostproc++; }
namespace { struct L5Printer { ~L5Printer(){ fprintf(stderr,"[L5INSTR] get_block_table n=%ld sum=%.2fms mean=%.4fms | set_inputs n=%ld sum=%.2fms mean=%.4fms | hostproc n=%ld sum=%.2fms mean=%.4fms\n", g_l5_n_gbt, g_l5_t_gbt/1e6, g_l5_n_gbt? g_l5_t_gbt/1e6/g_l5_n_gbt:0.0, g_l5_n_setinp, g_l5_t_setinp/1e6, g_l5_n_setinp? g_l5_t_setinp/1e6/g_l5_n_setinp:0.0, g_l5_n_hostproc, g_l5_t_hostproc/1e6, g_l5_n_hostproc? g_l5_t_hostproc/1e6/g_l5_n_hostproc:0.0 ); } } g_l5_printer; }


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

// Block table filler for the in-kernel paged read: fills an I32 [n_blk, n_stream]
// tensor with each stream's position-ordered cells, padded to n_blk (per column)
// with a masked empty cell, by delegating to the kv-cache context.
class input_block_table : public llm_graph_input_i {
public:
    input_block_table(const llama_kv_cache_context * mctx, ggml_tensor * idxs, uint32_t n_blk)
        : mctx(mctx), idxs(idxs), n_blk(n_blk) {}

    void set_input(const llama_ubatch * ubatch) override {
        GGML_UNUSED(ubatch);
        GGML_ASSERT(idxs && ggml_backend_buffer_is_host(idxs->buffer));
        double _t=l5_now_ns();
        mctx->get_block_table((int32_t *) idxs->data, n_blk);
        g_l5_t_gbt += l5_now_ns()-_t; g_l5_n_gbt++;
    }

    const llama_kv_cache_context * mctx;
    ggml_tensor * idxs;
    uint32_t n_blk;
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

bool in_kernel_decode(ggml_context * ctx0,
                      llm_graph_result * res,
                      const llama_kv_cache_context * mctx,
                      ggml_tensor ** k,
                      ggml_tensor ** v,
                      ggml_tensor ** kq_mask,
                      ggml_tensor ** block_table) {
    if (!active()) {
        return false;
    }
    // Bench escape hatch: LLAMA_KV_PAGED_GATHER=1 forces the old gather-read decode
    // path (for a same-build BEFORE/AFTER decode-step comparison). Dev-only.
    static const bool force_gather = (std::getenv("LLAMA_KV_PAGED_GATHER") != nullptr);
    if (force_gather) {
        return false;
    }

    ggml_tensor * K = *k;
    ggml_tensor * V = *v;
    ggml_tensor * M = *kq_mask;

    const int64_t n_stream = K->ne[3];
    GGML_ASSERT(M->ne[3] == n_stream);

    const int64_t n_gather = (int64_t) mctx->get_n_gather();
    if (n_gather <= 0) {
        // Worst-case reserve / nothing placed yet: keep the dense [0,n_kv) read.
        return false;
    }

    // The in-kernel read addresses V along its d-major (non-transposed) axis. If
    // the cache stores V transposed, fall back to gather() (which normalizes it).
    if (V->nb[1] > V->nb[2]) {
        return false;
    }

    if (debug()) {
        static int64_t once = 0;
        if (once++ < 2) {
            fprintf(stderr, "[paged-attn] in-kernel decode n_stream=%lld n_kv=%lld n_gather=%lld\n",
                    (long long) n_stream, (long long) K->ne[2], (long long) n_gather);
        }
    }

    // Block table [n_gather, n_stream]: column s holds stream s's non-empty cells
    // in token-POSITION order (identical to the gather index, so the reduction
    // order matches stock bit-for-bit), padded with a masked empty cell. Filled
    // at set_input from the kv-cache (get_gather_idxs), exactly like the gather.
    // Pad the logical length to FATTN_KQ_STRIDE (256) so the CUDA fattn vec kernel
    // reads fixed 128-wide KV blocks without overrun and the KV_max mask scan
    // engages; padded entries point at a masked empty cell (0 contribution). Stays
    // <= n_kv since n_kv is itself padded to 256 and n_gather <= n_kv.
    int64_t n_view = GGML_PAD(n_gather, 256);
    if (n_view > K->ne[2]) {
        n_view = K->ne[2];
    }

    // The flash-attn KV tile is 64 rows wide (nbatch_fa for head_dim 128). n_view must be
    // a whole number of such tiles so the in-kernel decode never reads past the gathered
    // rows: the trailing pad cells [n_gather, n_view) are all -inf, so any tile straddling
    // the boundary still contributes zero. This holds today only because the pad (256) is a
    // multiple of the tile; a future pad < 256 (or nbatch_fa > 256) that broke it would
    // silently reintroduce a past-end KV leak, so assert it rather than trust it.
    // pad must be a multiple of the flash-attn KV tile so the last tile is fully inside the -inf pad
    GGML_ASSERT(n_view % 64 == 0);

    ggml_tensor * idx = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_view, n_stream);
    ggml_set_input(idx);
    res->add_input(llm_graph_input_ptr(new input_block_table(mctx, idx, (uint32_t) n_view)));

    // Present K and V as [d, h, n_view, ns] VIEWS of the full physical window:
    // identical per-cell (nb1,nb2) and per-stream (nb3) strides, only the cell
    // dim shrinks to n_view. NOT materialized - the kernel reads in place.
    *k = ggml_view_4d(ctx0, K, K->ne[0], K->ne[1], n_view, n_stream,
                      K->nb[1], K->nb[2], K->nb[3], 0);
    *v = ggml_view_4d(ctx0, V, V->ne[0], V->ne[1], n_view, n_stream,
                      V->nb[1], V->nb[2], V->nb[3], 0);

    // Compact the mask to [n_gather, n_tps, 1, ns] in the same position order so
    // the kernel's logical mask index aligns with the block table. Cheap: the
    // mask is ~(d*h) smaller than K/V, which is why only its get_rows remains.
    {
        ggml_tensor * m = ggml_reshape_3d(ctx0, M, M->ne[0], M->ne[1], n_stream);
        m = ggml_cont(ctx0, ggml_transpose(ctx0, m));
        m = ggml_get_rows(ctx0, m, idx);
        m = ggml_cont(ctx0, ggml_transpose(ctx0, m));
        m = ggml_reshape_4d(ctx0, m, n_view, M->ne[1], 1, n_stream);
        if (M->type != m->type) {
            m = ggml_cast(ctx0, m, M->type);
        }
        *kq_mask = m;
    }

    *block_table = idx;
    return true;
}

} // namespace paged_attn
