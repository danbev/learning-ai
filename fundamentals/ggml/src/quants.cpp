#include <cstdio>
#include <vector>

#include "ggml.h"
#include "ggml-cpu.h"

int main(int argc, char **argv) {
    printf("GGML accumulate example\n");

    struct ggml_init_params params = {
    .mem_size   = 16*1024*1024,
    .mem_buffer = NULL,
    };
    struct ggml_context* ctx = ggml_init(params);

    ggml_quantize_init(GGML_TYPE_Q4_K);

    // So we need 8 blocks of 32 elements each, which is 256 elements total.
    float src[256] = {
        10.0, 11.0, 13.0, 14.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,

        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,

        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,

        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,

        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,

        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,

        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,

        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0};

    size_t dst_size = ggml_row_size(GGML_TYPE_Q4_K, 256) * 1;
    std::vector<uint8_t> dst(dst_size);

    auto ret = ggml_quantize_chunk(GGML_TYPE_Q4_K, src, dst.data(), 0, 1, 256, nullptr);

    fprintf(stderr, "Quantized %d elements to %zu bytes\n", 4, ret);
    for (size_t i = 0; i < dst.size(); ++i) {
        fprintf(stderr, "%02x ", dst[i]);
    }


    ggml_free(ctx);
    return 0;
}
