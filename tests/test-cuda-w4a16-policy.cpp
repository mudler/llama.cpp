#include "ggml-cuda/w4a16-policy.h"

#include <cstdio>
#include <cstdlib>

static void expect(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "%s\n", message);
        std::exit(1);
    }
}

static void test_direct_a_requires_master_w4a16() {
    const bool ok = ggml_cuda_w4a16_direct_a_should_engage_params(
        GGML_TYPE_NVFP4, GGML_TYPE_F32, GGML_TYPE_F32,
        /*blackwell=*/true,
        /*w4a16_prefill_m=*/0,
        /*direct_a=*/true,
        /*tokens=*/512,
        /*k=*/2048,
        /*n=*/512);
    expect(!ok, "direct-A engaged without LLAMA_W4A16_PREFILL_M");
}

static void test_direct_a_requires_direct_flag() {
    const bool ok = ggml_cuda_w4a16_direct_a_should_engage_params(
        GGML_TYPE_NVFP4, GGML_TYPE_F32, GGML_TYPE_F32,
        /*blackwell=*/true,
        /*w4a16_prefill_m=*/1,
        /*direct_a=*/false,
        /*tokens=*/512,
        /*k=*/2048,
        /*n=*/512);
    expect(!ok, "direct-A engaged without LLAMA_W4A16_DIRECT_A");
}

static void test_direct_a_engages_for_large_nvfp4_moe_prefill_shape() {
    const bool ok = ggml_cuda_w4a16_direct_a_should_engage_params(
        GGML_TYPE_NVFP4, GGML_TYPE_F32, GGML_TYPE_F32,
        /*blackwell=*/true,
        /*w4a16_prefill_m=*/1,
        /*direct_a=*/true,
        /*tokens=*/512,
        /*k=*/2048,
        /*n=*/512);
    expect(ok, "direct-A did not engage for a large NVFP4 MoE prefill shape");
}

static void test_direct_a_rejects_decode_sized_shape() {
    const bool ok = ggml_cuda_w4a16_direct_a_should_engage_params(
        GGML_TYPE_NVFP4, GGML_TYPE_F32, GGML_TYPE_F32,
        /*blackwell=*/true,
        /*w4a16_prefill_m=*/128,
        /*direct_a=*/true,
        /*tokens=*/128,
        /*k=*/2048,
        /*n=*/512);
    expect(!ok, "direct-A engaged for a decode-sized shape");
}

int main() {
    test_direct_a_requires_master_w4a16();
    test_direct_a_requires_direct_flag();
    test_direct_a_engages_for_large_nvfp4_moe_prefill_shape();
    test_direct_a_rejects_decode_sized_shape();
    std::puts("test-cuda-w4a16-policy: OK");
    return 0;
}
