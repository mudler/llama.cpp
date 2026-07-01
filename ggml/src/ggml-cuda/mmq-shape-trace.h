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
