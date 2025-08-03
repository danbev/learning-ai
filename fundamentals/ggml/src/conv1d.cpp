#include <stdio.h>
#include <string.h>

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

/*
 * A 1d convolution is a convolution operation that is applied to a
 * one-dimensional input, such as a sequence of numbers or a time series like
 * audio samples.
 *
 * But the input to this operation is not a 1d tensor.
 */
int main(int argc, char **argv) {
    printf("GGML conv_1d example\n\n");

    struct ggml_init_params params = {
    .mem_size   = 16*1024*1024,
    .mem_buffer = NULL,
    };
    struct ggml_context* ctx = ggml_init(params);

    struct ggml_tensor* kernel = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, 2);
    ggml_fp16_t kernel_data[] = { ggml_fp32_to_fp16(1.0f), ggml_fp32_to_fp16(2.0f) };

    ggml_set_name(kernel, "b");
    memcpy(kernel->data, kernel_data, ggml_nbytes(kernel));

    printf("kernel:\n");
    for (int y = 0; y < kernel->ne[0]; y++) {
      ggml_fp16_t* fp16_data = (ggml_fp16_t*) kernel->data;
      float converted_value = ggml_fp16_to_fp32(fp16_data[y]);
      printf("%.2f ", converted_value);
    }
    printf("\n");
    printf("\n");

    struct ggml_tensor* input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    float input_data[] = { 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f };
    ggml_set_name(input, "input");
    memcpy(input->data, input_data, ggml_nbytes(input));

    printf("input:\n");
    for (int y = 0; y < input->ne[0]; y++) {
        float* float_data = (float*)input->data;
        printf("%.2f ", float_data[y]);
    }
    printf("\n\n");

    int stride = 1;
    int pad = 0;
    int dil = 1;
    struct ggml_tensor* result = ggml_conv_1d(ctx, kernel, input, stride, pad, dil);

    struct ggml_cgraph* c_graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(c_graph, result);
    ggml_graph_compute_with_ctx(ctx, c_graph, 1);

    printf("result : type: %d, ne[0]: %ld\n", result->type, result->ne[0]);
    for (int y = 0; y < result->ne[0]; y++) {
        float* float_data = (float*)result->data;
        printf("%.2f ", float_data[y]);
    }
    printf("\n");

    ggml_free(ctx);
    return 0;
}
