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
        10.0, 11.0, 13.0, 14.0,
        10.0, 11.0, 13.0, 14.0,
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
        0.0, 0.0, 0.0, 0.0};

    std::vector<float> dst;

    auto ret = ggml_quantize_chunk(GGML_TYPE_Q4_K,
            src,
            dst.data(),
            0,
            4,
            256,
            nullptr);

    fprintf(stderr, "Quantized %d elements to %zu bytes\n", 4, ret);


    ggml_free(ctx);
    return 0;
}
