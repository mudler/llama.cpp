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

enum ggml_cuda_mul_mat_route {
    GGML_CUDA_MUL_MAT_ROUTE_FP4_PREFILL,
    GGML_CUDA_MUL_MAT_ROUTE_FWHT,
    GGML_CUDA_MUL_MAT_ROUTE_VEC_F,
    GGML_CUDA_MUL_MAT_ROUTE_MAT_F,
    GGML_CUDA_MUL_MAT_ROUTE_VEC_Q,
    GGML_CUDA_MUL_MAT_ROUTE_MMQ,
    GGML_CUDA_MUL_MAT_ROUTE_BATCHED_CUBLAS,
    GGML_CUDA_MUL_MAT_ROUTE_OP_VEC_F,
    GGML_CUDA_MUL_MAT_ROUTE_OP_VEC_Q,
    GGML_CUDA_MUL_MAT_ROUTE_OP_MMQ,
    GGML_CUDA_MUL_MAT_ROUTE_OP_CUBLAS,
};

enum ggml_cuda_cublas_route {
    GGML_CUDA_CUBLAS_ROUTE_NVFP4_BF16_TC,
    GGML_CUDA_CUBLAS_ROUTE_BF16_TC,
    GGML_CUDA_CUBLAS_ROUTE_F16_TC_32F,
    GGML_CUDA_CUBLAS_ROUTE_F16_TC_16F,
    GGML_CUDA_CUBLAS_ROUTE_SGEMM,
};

struct ggml_cuda_mul_mat_route_shape {
    ggml_cuda_mul_mat_route route;
    int type;
    int64_t ne1;
    int64_t ne11;
    int64_t ne12;
    int64_t ne13;
    bool split;
    bool use_vec_f;
    bool use_mat_f;
    bool use_vec_q;
    bool use_mmq;
    bool use_batched_cublas;
    bool use_fp4_prefill;
    bool use_fwht;
};

struct ggml_cuda_cublas_route_shape {
    ggml_cuda_cublas_route route;
    int type;
    int src1_type;
    int64_t row_diff;
    int64_t src1_ncols;
    int64_t ne00;
    int64_t ne10;
    int64_t ldc;
    bool supports_bf16;
    bool use_fp16;
    bool fast_fp16;
    bool force_fp32;
    bool force_fp16;
    bool src0_contig;
    bool full_rows;
    bool is_cdna;
    bool is_rdna4;
    bool is_volta;
    const char * src0_name;
    const char * src1_name;
    const char * dst_name;
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

static inline const char * ggml_cuda_mul_mat_route_name(const ggml_cuda_mul_mat_route route) {
    switch (route) {
        case GGML_CUDA_MUL_MAT_ROUTE_FP4_PREFILL:    return "fp4_prefill";
        case GGML_CUDA_MUL_MAT_ROUTE_FWHT:           return "fwht";
        case GGML_CUDA_MUL_MAT_ROUTE_VEC_F:          return "vec_f";
        case GGML_CUDA_MUL_MAT_ROUTE_MAT_F:          return "mat_f";
        case GGML_CUDA_MUL_MAT_ROUTE_VEC_Q:          return "vec_q";
        case GGML_CUDA_MUL_MAT_ROUTE_MMQ:            return "mmq";
        case GGML_CUDA_MUL_MAT_ROUTE_BATCHED_CUBLAS: return "batched_cublas";
        case GGML_CUDA_MUL_MAT_ROUTE_OP_VEC_F:       return "op_vec_f";
        case GGML_CUDA_MUL_MAT_ROUTE_OP_VEC_Q:       return "op_vec_q";
        case GGML_CUDA_MUL_MAT_ROUTE_OP_MMQ:         return "op_mmq";
        case GGML_CUDA_MUL_MAT_ROUTE_OP_CUBLAS:      return "op_cublas";
    }

    return "unknown";
}

static inline const char * ggml_cuda_cublas_route_name(const ggml_cuda_cublas_route route) {
    switch (route) {
        case GGML_CUDA_CUBLAS_ROUTE_NVFP4_BF16_TC: return "nvfp4_bf16_tc";
        case GGML_CUDA_CUBLAS_ROUTE_BF16_TC:       return "bf16_tc";
        case GGML_CUDA_CUBLAS_ROUTE_F16_TC_32F:    return "f16_tc_32f";
        case GGML_CUDA_CUBLAS_ROUTE_F16_TC_16F:    return "f16_tc_16f";
        case GGML_CUDA_CUBLAS_ROUTE_SGEMM:         return "sgemm";
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

static inline ggml_cuda_mul_mat_route_shape ggml_cuda_mul_mat_route_shape_make(
        const int type, const int64_t ne1, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const bool split, const bool use_vec_f, const bool use_mat_f, const bool use_vec_q,
        const bool use_mmq, const bool use_batched_cublas, const bool use_fp4_prefill,
        const bool use_fwht) {
    ggml_cuda_mul_mat_route route = GGML_CUDA_MUL_MAT_ROUTE_OP_CUBLAS;
    if (use_fp4_prefill) {
        route = GGML_CUDA_MUL_MAT_ROUTE_FP4_PREFILL;
    } else if (use_fwht) {
        route = GGML_CUDA_MUL_MAT_ROUTE_FWHT;
    } else if (!split && use_vec_f) {
        route = GGML_CUDA_MUL_MAT_ROUTE_VEC_F;
    } else if (!split && use_mat_f) {
        route = GGML_CUDA_MUL_MAT_ROUTE_MAT_F;
    } else if (!split && use_vec_q) {
        route = GGML_CUDA_MUL_MAT_ROUTE_VEC_Q;
    } else if (!split && use_mmq) {
        route = GGML_CUDA_MUL_MAT_ROUTE_MMQ;
    } else if (!split && use_batched_cublas) {
        route = GGML_CUDA_MUL_MAT_ROUTE_BATCHED_CUBLAS;
    } else if (use_vec_f) {
        route = GGML_CUDA_MUL_MAT_ROUTE_OP_VEC_F;
    } else if (use_vec_q) {
        route = GGML_CUDA_MUL_MAT_ROUTE_OP_VEC_Q;
    } else if (use_mmq) {
        route = GGML_CUDA_MUL_MAT_ROUTE_OP_MMQ;
    }

    return {
        route,
        type,
        ne1,
        ne11,
        ne12,
        ne13,
        split,
        use_vec_f,
        use_mat_f,
        use_vec_q,
        use_mmq,
        use_batched_cublas,
        use_fp4_prefill,
        use_fwht,
    };
}

static inline ggml_cuda_cublas_route_shape ggml_cuda_cublas_route_shape_make(
        const int type, const int src1_type, const int64_t row_diff, const int64_t src1_ncols,
        const int64_t ne00, const int64_t ne10, const int64_t ldc, const bool supports_bf16,
        const bool use_fp16, const bool fast_fp16, const bool force_fp32, const bool force_fp16,
        const bool src0_contig, const bool full_rows, const bool is_cdna, const bool is_rdna4,
        const bool is_volta, const char * src0_name, const char * src1_name, const char * dst_name) {
    ggml_cuda_cublas_route route = GGML_CUDA_CUBLAS_ROUTE_SGEMM;
    if (supports_bf16 && type == 40 && src0_contig && full_rows) {
        route = GGML_CUDA_CUBLAS_ROUTE_NVFP4_BF16_TC;
    } else if (supports_bf16 && type == 30 && src0_contig && full_rows) {
        route = GGML_CUDA_CUBLAS_ROUTE_BF16_TC;
    } else if (fast_fp16 && use_fp16) {
        if (!force_fp16 && (is_cdna || is_rdna4 || is_volta || force_fp32)) {
            route = GGML_CUDA_CUBLAS_ROUTE_F16_TC_32F;
        } else {
            route = GGML_CUDA_CUBLAS_ROUTE_F16_TC_16F;
        }
    }

    return {
        route,
        type,
        src1_type,
        row_diff,
        src1_ncols,
        ne00,
        ne10,
        ldc,
        supports_bf16,
        use_fp16,
        fast_fp16,
        force_fp32,
        force_fp16,
        src0_contig,
        full_rows,
        is_cdna,
        is_rdna4,
        is_volta,
        src0_name ? src0_name : "",
        src1_name ? src1_name : "",
        dst_name ? dst_name : "",
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

static inline int ggml_cuda_mul_mat_route_shape_format(
        char * buf, const size_t size, const ggml_cuda_mul_mat_route_shape & shape) {
    return std::snprintf(buf, size,
        "route=%s type=%d ne1=%lld ne11=%lld ne12=%lld ne13=%lld split=%d "
        "use_vec_f=%d use_mat_f=%d use_vec_q=%d use_mmq=%d use_batched_cublas=%d "
        "use_fp4_prefill=%d use_fwht=%d",
        ggml_cuda_mul_mat_route_name(shape.route),
        shape.type,
        (long long) shape.ne1,
        (long long) shape.ne11,
        (long long) shape.ne12,
        (long long) shape.ne13,
        shape.split ? 1 : 0,
        shape.use_vec_f ? 1 : 0,
        shape.use_mat_f ? 1 : 0,
        shape.use_vec_q ? 1 : 0,
        shape.use_mmq ? 1 : 0,
        shape.use_batched_cublas ? 1 : 0,
        shape.use_fp4_prefill ? 1 : 0,
        shape.use_fwht ? 1 : 0);
}

static inline int ggml_cuda_cublas_route_shape_format(
        char * buf, const size_t size, const ggml_cuda_cublas_route_shape & shape) {
    return std::snprintf(buf, size,
        "route=%s type=%d src1_type=%d row_diff=%lld src1_ncols=%lld ne00=%lld ne10=%lld ldc=%lld "
        "supports_bf16=%d use_fp16=%d fast_fp16=%d force_fp32=%d force_fp16=%d "
        "src0_contig=%d full_rows=%d is_cdna=%d is_rdna4=%d is_volta=%d "
        "src0=%s src1=%s dst=%s",
        ggml_cuda_cublas_route_name(shape.route),
        shape.type,
        shape.src1_type,
        (long long) shape.row_diff,
        (long long) shape.src1_ncols,
        (long long) shape.ne00,
        (long long) shape.ne10,
        (long long) shape.ldc,
        shape.supports_bf16 ? 1 : 0,
        shape.use_fp16 ? 1 : 0,
        shape.fast_fp16 ? 1 : 0,
        shape.force_fp32 ? 1 : 0,
        shape.force_fp16 ? 1 : 0,
        shape.src0_contig ? 1 : 0,
        shape.full_rows ? 1 : 0,
        shape.is_cdna ? 1 : 0,
        shape.is_rdna4 ? 1 : 0,
        shape.is_volta ? 1 : 0,
        shape.src0_name,
        shape.src1_name,
        shape.dst_name);
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
