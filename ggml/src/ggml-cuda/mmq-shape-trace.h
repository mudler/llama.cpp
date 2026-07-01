#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>

struct ggml_cuda_mmq_shape {
    int type;
    bool is_moe;
    int64_t ncols_dst;
    int64_t nchannels_x;
    int64_t ncols_max;
    int64_t n_active_est;
    int64_t density;
    int mmq_x_max;
    int mmq_x_lim;
    int mmq_x_best;
    int mmq_y;
    bool use_stream_k;
};

struct ggml_cuda_mmq_launch_shape {
    int type;
    bool is_moe;
    int64_t ncols_dst;
    int64_t ncols_max;
    int mmq_x;
    int mmq_y;
    int ntx;
    int nty;
    int ntzw;
    int ntiles_dst;
    int nsm;
    int tiles_nwaves;
    int tiles_efficiency_percent;
    int stream_k_blocks;
    bool fixup_needed;
};

struct ggml_cuda_mmq_small_m_shape {
    bool is_moe;
    int64_t ncols_dst;
    int64_t nchannels_x;
    int64_t ncols_max;
    int64_t n_active_est;
    int64_t density;
    int mmq_x_best;
    bool use_stream_k;
    bool is_candidate;
};

enum ggml_cuda_mmid_route {
    GGML_CUDA_MMID_ROUTE_MMVQ,
    GGML_CUDA_MMID_ROUTE_MMVF,
    GGML_CUDA_MMID_ROUTE_MMQ,
    GGML_CUDA_MMID_ROUTE_MMF,
    GGML_CUDA_MMID_ROUTE_FALLBACK,
};

struct ggml_cuda_mmid_route_shape {
    ggml_cuda_mmid_route route;
    int type;
    int64_t ne2;
    int64_t ne12;
    int64_t n_experts;
    int mmvq_max;
    bool use_mmq;
    bool use_mmf;
    bool is_amd;
    bool is_quantized;
    bool host_sync;
};

static inline const char * ggml_cuda_mmid_route_name(const ggml_cuda_mmid_route route) {
    switch (route) {
        case GGML_CUDA_MMID_ROUTE_MMVQ:     return "mmvq";
        case GGML_CUDA_MMID_ROUTE_MMVF:     return "mmvf";
        case GGML_CUDA_MMID_ROUTE_MMQ:      return "mmq";
        case GGML_CUDA_MMID_ROUTE_MMF:      return "mmf";
        case GGML_CUDA_MMID_ROUTE_FALLBACK: return "fallback";
    }

    return "unknown";
}

static inline ggml_cuda_mmq_shape ggml_cuda_mmq_shape_make(
        const int type, const bool is_moe, const int64_t ncols_dst, const int64_t nchannels_x,
        const int64_t ncols_max, const int mmq_x_max, const int mmq_x_lim, const int mmq_x_best,
        const int mmq_y, const bool use_stream_k) {
    int64_t n_active_est = 0;
    int64_t density = 0;
    if (is_moe && ncols_dst > 0 && nchannels_x > 0) {
        n_active_est = ncols_dst < nchannels_x ? ncols_dst : nchannels_x;
        density = (ncols_dst + n_active_est - 1) / n_active_est;
    }

    return {
        type,
        is_moe,
        ncols_dst,
        nchannels_x,
        ncols_max,
        n_active_est,
        density,
        mmq_x_max,
        mmq_x_lim,
        mmq_x_best,
        mmq_y,
        use_stream_k,
    };
}

static inline ggml_cuda_mmid_route_shape ggml_cuda_mmid_route_shape_make(
        const int type, const int64_t ne2, const int64_t ne12, const int64_t n_experts,
        const int mmvq_max, const bool use_mmq, const bool use_mmf, const bool is_amd,
        const bool is_quantized) {
    ggml_cuda_mmid_route route = GGML_CUDA_MMID_ROUTE_FALLBACK;
    if (ne2 <= mmvq_max) {
        if (is_quantized) {
            route = GGML_CUDA_MMID_ROUTE_MMVQ;
        } else if (is_amd) {
            route = GGML_CUDA_MMID_ROUTE_MMVF;
        }
    }

    if (route == GGML_CUDA_MMID_ROUTE_FALLBACK) {
        if (use_mmq) {
            route = GGML_CUDA_MMID_ROUTE_MMQ;
        } else if (use_mmf) {
            route = GGML_CUDA_MMID_ROUTE_MMF;
        }
    }

    return {
        route,
        type,
        ne2,
        ne12,
        n_experts,
        mmvq_max,
        use_mmq,
        use_mmf,
        is_amd,
        is_quantized,
        route == GGML_CUDA_MMID_ROUTE_FALLBACK,
    };
}

static inline ggml_cuda_mmq_small_m_shape ggml_cuda_mmq_small_m_shape_make(
        const bool is_moe, const int64_t ncols_dst, const int64_t nchannels_x,
        const int64_t ncols_max, const int mmq_x_best, const bool use_stream_k) {
    int64_t n_active_est = 0;
    int64_t density = 0;
    if (is_moe && ncols_dst > 0 && nchannels_x > 0) {
        n_active_est = ncols_dst < nchannels_x ? ncols_dst : nchannels_x;
        density = (ncols_dst + n_active_est - 1) / n_active_est;
    }

    const bool is_candidate =
        is_moe &&
        use_stream_k &&
        ncols_max > 0 && ncols_max <= 128 &&
        density > 0 && density <= 4 &&
        mmq_x_best > 0 && mmq_x_best <= 64;

    return {
        is_moe,
        ncols_dst,
        nchannels_x,
        ncols_max,
        n_active_est,
        density,
        mmq_x_best,
        use_stream_k,
        is_candidate,
    };
}

static inline ggml_cuda_mmq_launch_shape ggml_cuda_mmq_launch_shape_make(
        const int type, const bool is_moe, const int64_t ncols_dst, const int64_t ncols_max,
        const int mmq_x, const int mmq_y, const int ntx, const int nty, const int ntzw,
        const int ntiles_dst, const int nsm, const int tiles_nwaves, const int tiles_efficiency_percent,
        const int stream_k_blocks, const bool fixup_needed) {
    return {
        type,
        is_moe,
        ncols_dst,
        ncols_max,
        mmq_x,
        mmq_y,
        ntx,
        nty,
        ntzw,
        ntiles_dst,
        nsm,
        tiles_nwaves,
        tiles_efficiency_percent,
        stream_k_blocks,
        fixup_needed,
    };
}

static inline int ggml_cuda_mmq_shape_format(char * buf, const size_t size, const ggml_cuda_mmq_shape & shape) {
    return std::snprintf(buf, size,
        "type=%d moe=%d ncols_dst=%lld nchannels_x=%lld ncols_max=%lld "
        "n_active_est=%lld density=%lld mmq_x_max=%d mmq_x_lim=%d "
        "mmq_x_best=%d mmq_y=%d stream_k=%d",
        shape.type,
        shape.is_moe ? 1 : 0,
        (long long) shape.ncols_dst,
        (long long) shape.nchannels_x,
        (long long) shape.ncols_max,
        (long long) shape.n_active_est,
        (long long) shape.density,
        shape.mmq_x_max,
        shape.mmq_x_lim,
        shape.mmq_x_best,
        shape.mmq_y,
        shape.use_stream_k ? 1 : 0);
}

static inline int ggml_cuda_mmq_launch_shape_format(
        char * buf, const size_t size, const ggml_cuda_mmq_launch_shape & shape) {
    return std::snprintf(buf, size,
        "type=%d moe=%d ncols_dst=%lld ncols_max=%lld mmq_x=%d mmq_y=%d "
        "ntx=%d nty=%d ntzw=%d ntiles_dst=%d nsm=%d tiles_nwaves=%d "
        "tiles_efficiency=%d stream_k_blocks=%d fixup=%d",
        shape.type,
        shape.is_moe ? 1 : 0,
        (long long) shape.ncols_dst,
        (long long) shape.ncols_max,
        shape.mmq_x,
        shape.mmq_y,
        shape.ntx,
        shape.nty,
        shape.ntzw,
        shape.ntiles_dst,
        shape.nsm,
        shape.tiles_nwaves,
        shape.tiles_efficiency_percent,
        shape.stream_k_blocks,
        shape.fixup_needed ? 1 : 0);
}

static inline int ggml_cuda_mmid_route_shape_format(
        char * buf, const size_t size, const ggml_cuda_mmid_route_shape & shape) {
    return std::snprintf(buf, size,
        "route=%s type=%d host_sync=%d ne2=%lld ne12=%lld n_experts=%lld "
        "mmvq_max=%d use_mmq=%d use_mmf=%d is_amd=%d is_quantized=%d",
        ggml_cuda_mmid_route_name(shape.route),
        shape.type,
        shape.host_sync ? 1 : 0,
        (long long) shape.ne2,
        (long long) shape.ne12,
        (long long) shape.n_experts,
        shape.mmvq_max,
        shape.use_mmq ? 1 : 0,
        shape.use_mmf ? 1 : 0,
        shape.is_amd ? 1 : 0,
        shape.is_quantized ? 1 : 0);
}

static inline int ggml_cuda_mmq_small_m_shape_format(
        char * buf, const size_t size, const ggml_cuda_mmq_small_m_shape & shape) {
    return std::snprintf(buf, size,
        "candidate=%d moe=%d ncols_dst=%lld nchannels_x=%lld ncols_max=%lld "
        "n_active_est=%lld density=%lld mmq_x_best=%d stream_k=%d",
        shape.is_candidate ? 1 : 0,
        shape.is_moe ? 1 : 0,
        (long long) shape.ncols_dst,
        (long long) shape.nchannels_x,
        (long long) shape.ncols_max,
        (long long) shape.n_active_est,
        (long long) shape.density,
        shape.mmq_x_best,
        shape.use_stream_k ? 1 : 0);
}

static inline int ggml_cuda_mmq_small_m_tile_limit(
        const ggml_cuda_mmq_small_m_shape & shape, const int current_limit, const int requested_tile) {
    if (!shape.is_candidate || requested_tile < 8 || requested_tile >= current_limit) {
        return current_limit;
    }

    return requested_tile;
}
