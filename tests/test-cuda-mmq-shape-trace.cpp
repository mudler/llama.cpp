#include "ggml-cuda/mmq-shape-trace.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

static void require(bool ok, const char * what) {
    if (!ok) {
        std::fprintf(stderr, "require failed: %s\n", what);
        std::exit(1);
    }
}

int main() {
    const ggml_cuda_mmq_shape shape = ggml_cuda_mmq_shape_make(
        /* type */ 39,
        /* is_moe */ true,
        /* ncols_dst */ 1024,
        /* nchannels_x */ 256,
        /* ncols_max */ 128,
        /* mmq_x_max */ 128,
        /* mmq_x_lim */ 64,
        /* mmq_x_best */ 64,
        /* mmq_y */ 128,
        /* use_stream_k */ true);

    require(shape.n_active_est == 256, "active expert estimate is capped by expert count");
    require(shape.density == 4, "density is ceil(assignments / active experts)");

    char buf[256];
    const int n = ggml_cuda_mmq_shape_format(buf, sizeof(buf), shape);

    require(n > 0, "format returns byte count");
    require(std::strstr(buf, "moe=1") != nullptr, "trace includes moe flag");
    require(std::strstr(buf, "ncols_dst=1024") != nullptr, "trace includes routed assignment count");
    require(std::strstr(buf, "n_active_est=256") != nullptr, "trace includes active estimate");
    require(std::strstr(buf, "density=4") != nullptr, "trace includes density");
    require(std::strstr(buf, "mmq_x_best=64") != nullptr, "trace includes selected tile");
    require(std::strstr(buf, "stream_k=1") != nullptr, "trace includes stream-k flag");

    const ggml_cuda_mmq_launch_shape launch = ggml_cuda_mmq_launch_shape_make(
        /* type */ 39,
        /* is_moe */ true,
        /* ncols_dst */ 1024,
        /* ncols_max */ 128,
        /* mmq_x */ 64,
        /* mmq_y */ 128,
        /* ntx */ 2,
        /* nty */ 4,
        /* ntzw */ 3,
        /* ntiles_dst */ 24,
        /* nsm */ 16,
        /* tiles_nwaves */ 2,
        /* tiles_efficiency_percent */ 75,
        /* stream_k_blocks */ 16,
        /* fixup_needed */ true);

    const int launch_n = ggml_cuda_mmq_launch_shape_format(buf, sizeof(buf), launch);

    require(launch_n > 0, "launch format returns byte count");
    require(std::strstr(buf, "moe=1") != nullptr, "launch trace includes moe flag");
    require(std::strstr(buf, "ncols_dst=1024") != nullptr, "launch trace includes routed assignment count");
    require(std::strstr(buf, "ncols_max=128") != nullptr, "launch trace includes max column count");
    require(std::strstr(buf, "mmq_x=64") != nullptr, "launch trace includes compiled x tile");
    require(std::strstr(buf, "mmq_y=128") != nullptr, "launch trace includes compiled y tile");
    require(std::strstr(buf, "ntx=2") != nullptr, "launch trace includes x tile count");
    require(std::strstr(buf, "nty=4") != nullptr, "launch trace includes y tile count");
    require(std::strstr(buf, "ntzw=3") != nullptr, "launch trace includes batch tile count");
    require(std::strstr(buf, "ntiles_dst=24") != nullptr, "launch trace includes total tile count");
    require(std::strstr(buf, "nsm=16") != nullptr, "launch trace includes SM count");
    require(std::strstr(buf, "tiles_nwaves=2") != nullptr, "launch trace includes wave count");
    require(std::strstr(buf, "tiles_efficiency=75") != nullptr, "launch trace includes stream-k efficiency");
    require(std::strstr(buf, "stream_k_blocks=16") != nullptr, "launch trace includes actual stream-k block count");
    require(std::strstr(buf, "fixup=1") != nullptr, "launch trace includes fixup flag");

    const ggml_cuda_mmq_small_m_shape small_m = ggml_cuda_mmq_small_m_shape_make(
        /* is_moe */ true,
        /* ncols_dst */ 1024,
        /* nchannels_x */ 256,
        /* ncols_max */ 128,
        /* mmq_x_best */ 64,
        /* use_stream_k */ true);

    require(small_m.is_candidate, "decode-like low-density MoE shape is a small-M candidate");
    require(small_m.n_active_est == 256, "small-M active estimate is capped by expert count");
    require(small_m.density == 4, "small-M density is ceil(assignments / active experts)");

    require(!ggml_cuda_mmq_small_m_shape_make(
        /* is_moe */ false, 1024, 256, 128, 64, true).is_candidate,
        "dense shape is excluded");
    require(!ggml_cuda_mmq_small_m_shape_make(
        /* is_moe */ true, 4096, 256, 512, 128, true).is_candidate,
        "prefill-like shape is excluded");
    require(!ggml_cuda_mmq_small_m_shape_make(
        /* is_moe */ true, 4096, 256, 128, 64, true).is_candidate,
        "high-density shape is excluded");
    require(!ggml_cuda_mmq_small_m_shape_make(
        /* is_moe */ true, 1024, 256, 128, 128, true).is_candidate,
        "large selected tile is excluded");
    require(!ggml_cuda_mmq_small_m_shape_make(
        /* is_moe */ true, 1024, 256, 128, 64, false).is_candidate,
        "non-stream-k shape is excluded");

    const int small_m_n = ggml_cuda_mmq_small_m_shape_format(buf, sizeof(buf), small_m);

    require(small_m_n > 0, "small-M format returns byte count");
    require(std::strstr(buf, "candidate=1") != nullptr, "small-M trace includes candidate flag");
    require(std::strstr(buf, "density=4") != nullptr, "small-M trace includes density");
    require(std::strstr(buf, "mmq_x_best=64") != nullptr, "small-M trace includes selected tile");

    require(ggml_cuda_mmq_small_m_tile_limit(small_m, 64, 0) == 64,
        "small-M tile override is default-off");
    require(ggml_cuda_mmq_small_m_tile_limit(small_m, 64, 16) == 16,
        "small-M tile override caps candidate tile limit");
    require(ggml_cuda_mmq_small_m_tile_limit(small_m, 64, 128) == 64,
        "small-M tile override ignores non-smaller tiles");
    require(ggml_cuda_mmq_small_m_tile_limit(
        ggml_cuda_mmq_small_m_shape_make(/* is_moe */ true, 4096, 256, 512, 128, true), 128, 16) == 128,
        "small-M tile override excludes prefill-like shapes");

    return 0;
}
