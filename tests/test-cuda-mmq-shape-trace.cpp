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

    const ggml_cuda_mmid_route_shape mmvq = ggml_cuda_mmid_route_shape_make(
        /* type */ 39,
        /* ne2 */ 1,
        /* ne12 */ 1,
        /* ne02 */ 256,
        /* mmvq_max */ 4,
        /* use_mmq */ true,
        /* use_mmf */ true,
        /* is_amd */ false,
        /* is_quantized */ true);

    require(mmvq.route == GGML_CUDA_MMID_ROUTE_MMVQ, "MMVQ wins when within its batch cap");
    require(!mmvq.host_sync, "MMVQ route is graph-safe");

    const ggml_cuda_mmid_route_shape mmq = ggml_cuda_mmid_route_shape_make(
        /* type */ 39,
        /* ne2 */ 128,
        /* ne12 */ 128,
        /* ne02 */ 256,
        /* mmvq_max */ 4,
        /* use_mmq */ true,
        /* use_mmf */ true,
        /* is_amd */ false,
        /* is_quantized */ true);

    require(mmq.route == GGML_CUDA_MMID_ROUTE_MMQ, "grouped MMQ wins after MMVQ cap");
    require(!mmq.host_sync, "grouped MMQ route is graph-safe");

    const ggml_cuda_mmid_route_shape mmf = ggml_cuda_mmid_route_shape_make(
        /* type */ 1,
        /* ne2 */ 128,
        /* ne12 */ 128,
        /* ne02 */ 256,
        /* mmvq_max */ 4,
        /* use_mmq */ false,
        /* use_mmf */ true,
        /* is_amd */ false,
        /* is_quantized */ false);

    require(mmf.route == GGML_CUDA_MMID_ROUTE_MMF, "MMF wins when grouped MMQ is unavailable");
    require(!mmf.host_sync, "MMF route is graph-safe");

    const ggml_cuda_mmid_route_shape fallback = ggml_cuda_mmid_route_shape_make(
        /* type */ 0,
        /* ne2 */ 128,
        /* ne12 */ 128,
        /* ne02 */ 256,
        /* mmvq_max */ 4,
        /* use_mmq */ false,
        /* use_mmf */ false,
        /* is_amd */ false,
        /* is_quantized */ false);

    require(fallback.route == GGML_CUDA_MMID_ROUTE_FALLBACK, "fallback is used when device routes do not match");
    require(fallback.host_sync, "fallback route requires host synchronization");

    const ggml_cuda_mmid_route_shape amd_mmvf = ggml_cuda_mmid_route_shape_make(
        /* type */ 1,
        /* ne2 */ 1,
        /* ne12 */ 1,
        /* ne02 */ 256,
        /* mmvq_max */ 4,
        /* use_mmq */ false,
        /* use_mmf */ false,
        /* is_amd */ true,
        /* is_quantized */ false);

    require(amd_mmvf.route == GGML_CUDA_MMID_ROUTE_MMVF, "AMD float vector route wins within MMVQ cap");
    require(!amd_mmvf.host_sync, "AMD float vector route is graph-safe");

    const int route_n = ggml_cuda_mmid_route_shape_format(buf, sizeof(buf), mmq);

    require(route_n > 0, "MMID route format returns byte count");
    require(std::strstr(buf, "route=mmq") != nullptr, "MMID trace includes route name");
    require(std::strstr(buf, "host_sync=0") != nullptr, "MMID trace includes host sync flag");
    require(std::strstr(buf, "ne2=128") != nullptr, "MMID trace includes destination batch");
    require(std::strstr(buf, "ne12=128") != nullptr, "MMID trace includes routed token count");
    require(std::strstr(buf, "n_experts=256") != nullptr, "MMID trace includes expert count");
    require(std::strstr(buf, "mmvq_max=4") != nullptr, "MMID trace includes MMVQ cap");
    require(std::strstr(buf, "use_mmq=1") != nullptr, "MMID trace includes MMQ predicate");
    require(std::strstr(buf, "use_mmf=1") != nullptr, "MMID trace includes MMF predicate");

    const ggml_cuda_mul_mat_route_shape mat_f = ggml_cuda_mul_mat_route_shape_make(
        /* type */ 30,
        /* ne1 */ 128,
        /* ne11 */ 128,
        /* ne12 */ 1,
        /* ne13 */ 1,
        /* split */ false,
        /* use_vec_f */ false,
        /* use_mat_f */ true,
        /* use_vec_q */ false,
        /* use_mmq */ false,
        /* use_batched_cublas */ false,
        /* use_fp4_prefill */ false,
        /* use_fwht */ false);

    require(mat_f.route == GGML_CUDA_MUL_MAT_ROUTE_MAT_F, "regular MUL_MAT prefers direct mat_f when available");
    require(!mat_f.split, "regular MUL_MAT trace records split flag");

    const ggml_cuda_mul_mat_route_shape batched = ggml_cuda_mul_mat_route_shape_make(
        /* type */ 31,
        /* ne1 */ 128,
        /* ne11 */ 128,
        /* ne12 */ 4,
        /* ne13 */ 1,
        /* split */ false,
        /* use_vec_f */ false,
        /* use_mat_f */ false,
        /* use_vec_q */ false,
        /* use_mmq */ false,
        /* use_batched_cublas */ true,
        /* use_fp4_prefill */ false,
        /* use_fwht */ false);

    require(batched.route == GGML_CUDA_MUL_MAT_ROUTE_BATCHED_CUBLAS,
        "regular MUL_MAT records batched cuBLAS route");

    const ggml_cuda_mul_mat_route_shape op_cublas = ggml_cuda_mul_mat_route_shape_make(
        /* type */ 0,
        /* ne1 */ 16,
        /* ne11 */ 16,
        /* ne12 */ 1,
        /* ne13 */ 1,
        /* split */ true,
        /* use_vec_f */ false,
        /* use_mat_f */ false,
        /* use_vec_q */ false,
        /* use_mmq */ false,
        /* use_batched_cublas */ false,
        /* use_fp4_prefill */ false,
        /* use_fwht */ false);

    require(op_cublas.route == GGML_CUDA_MUL_MAT_ROUTE_OP_CUBLAS,
        "regular MUL_MAT fallback records op cublas route");

    const ggml_cuda_mul_mat_route_shape fp4_prefill = ggml_cuda_mul_mat_route_shape_make(
        /* type */ 39,
        /* ne1 */ 512,
        /* ne11 */ 512,
        /* ne12 */ 1,
        /* ne13 */ 1,
        /* split */ false,
        /* use_vec_f */ false,
        /* use_mat_f */ false,
        /* use_vec_q */ false,
        /* use_mmq */ true,
        /* use_batched_cublas */ false,
        /* use_fp4_prefill */ true,
        /* use_fwht */ false);

    require(fp4_prefill.route == GGML_CUDA_MUL_MAT_ROUTE_FP4_PREFILL,
        "regular MUL_MAT records native FP4 prefill route before MMQ");

    const int mul_mat_route_n = ggml_cuda_mul_mat_route_shape_format(buf, sizeof(buf), mat_f);

    require(mul_mat_route_n > 0, "regular MUL_MAT route format returns byte count");
    require(std::strstr(buf, "route=mat_f") != nullptr, "regular MUL_MAT trace includes route name");
    require(std::strstr(buf, "type=30") != nullptr, "regular MUL_MAT trace includes type");
    require(std::strstr(buf, "ne1=128") != nullptr, "regular MUL_MAT trace includes output columns");
    require(std::strstr(buf, "ne11=128") != nullptr, "regular MUL_MAT trace includes src1 columns");
    require(std::strstr(buf, "ne12=1") != nullptr, "regular MUL_MAT trace includes src1 batch dim");
    require(std::strstr(buf, "split=0") != nullptr, "regular MUL_MAT trace includes split flag");
    require(std::strstr(buf, "use_mat_f=1") != nullptr, "regular MUL_MAT trace includes mat_f predicate");
    require(std::strstr(buf, "use_batched_cublas=0") != nullptr,
        "regular MUL_MAT trace includes batched cuBLAS predicate");

    const ggml_cuda_cublas_route_shape bf16_tc = ggml_cuda_cublas_route_shape_make(
        /* type */ 30,
        /* src1_type */ 0,
        /* row_diff */ 18,
        /* src1_ncols */ 18,
        /* ne00 */ 1024,
        /* ne10 */ 1024,
        /* ldc */ 18,
        /* supports_bf16 */ true,
        /* use_fp16 */ false,
        /* fast_fp16 */ true,
        /* force_fp32 */ false,
        /* force_fp16 */ false,
        /* src0_contig */ true,
        /* full_rows */ true,
        /* is_cdna */ false,
        /* is_rdna4 */ false,
        /* is_volta */ false);

    require(bf16_tc.route == GGML_CUDA_CUBLAS_ROUTE_BF16_TC,
        "cuBLAS records native BF16 tensor-core route");

    const ggml_cuda_cublas_route_shape nvfp4_bf16_tc = ggml_cuda_cublas_route_shape_make(
        /* type */ 40, 0, 128, 128, 1024, 1024, 128, true, false, true, false, false, true, true,
        false, false, false);

    require(nvfp4_bf16_tc.route == GGML_CUDA_CUBLAS_ROUTE_NVFP4_BF16_TC,
        "cuBLAS records NVFP4 dequant-to-BF16 tensor-core route");

    const ggml_cuda_cublas_route_shape f16_tc_16f = ggml_cuda_cublas_route_shape_make(
        /* type */ 1, 0, 64, 64, 1024, 1024, 64, false, true, true, false, false, true, true,
        false, false, false);

    require(f16_tc_16f.route == GGML_CUDA_CUBLAS_ROUTE_F16_TC_16F,
        "cuBLAS records default FP16 tensor-core 16F compute route");

    const ggml_cuda_cublas_route_shape f16_tc_32f = ggml_cuda_cublas_route_shape_make(
        /* type */ 1, 0, 64, 64, 1024, 1024, 64, false, true, true, true, false, true, true,
        false, false, false);

    require(f16_tc_32f.route == GGML_CUDA_CUBLAS_ROUTE_F16_TC_32F,
        "cuBLAS records forced FP16 tensor-core 32F compute route");

    const ggml_cuda_cublas_route_shape sgemm = ggml_cuda_cublas_route_shape_make(
        /* type */ 0, 0, 12, 12, 1024, 1024, 12, false, false, true, false, false, true, true,
        false, false, false);

    require(sgemm.route == GGML_CUDA_CUBLAS_ROUTE_SGEMM,
        "cuBLAS records SGEMM fallback route");

    const int cublas_route_n = ggml_cuda_cublas_route_shape_format(buf, sizeof(buf), bf16_tc);

    require(cublas_route_n > 0, "cuBLAS route format returns byte count");
    require(std::strstr(buf, "route=bf16_tc") != nullptr, "cuBLAS trace includes route name");
    require(std::strstr(buf, "type=30") != nullptr, "cuBLAS trace includes src0 type");
    require(std::strstr(buf, "src1_type=0") != nullptr, "cuBLAS trace includes src1 type");
    require(std::strstr(buf, "row_diff=18") != nullptr, "cuBLAS trace includes row count");
    require(std::strstr(buf, "src1_ncols=18") != nullptr, "cuBLAS trace includes source column count");
    require(std::strstr(buf, "supports_bf16=1") != nullptr, "cuBLAS trace includes BF16 predicate");
    require(std::strstr(buf, "force_fp32=0") != nullptr, "cuBLAS trace includes forced compute predicate");
    require(std::strstr(buf, "src0_contig=1") != nullptr, "cuBLAS trace includes contiguity predicate");

    return 0;
}
