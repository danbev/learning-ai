#include <stdio.h>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

ggml_backend_meta_split_state get_split_state(const struct ggml_tensor * tensor, void * user_data) {
    ggml_backend_meta_split_state state;

    // Replicate the tensor on all devices.
    state.axis = GGML_BACKEND_SPLIT_AXIS_MIRRORED;
    state.n_segments = 1;
    return state;
}

int main(int argc, char ** argv) {
    printf("GGML backend meta device example\n");

    struct ggml_init_params params = {
        .mem_size   = 16*1024*1024,
        .mem_buffer = NULL,
        .no_alloc = true,
    };

    struct ggml_context * ctx = ggml_init(params);

    ggml_backend_dev_t cuda_dev = ggml_backend_dev_by_name("CUDA0");
    if (!cuda_dev) {
        fprintf(stderr, "CUDA device not found\n");
        ggml_free(ctx);
        return 1;
    }

    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_name("CPU");
    if (!cuda_dev) {
        fprintf(stderr, "CPU device not found\n");
        ggml_free(ctx);
        return 1;
    }

    ggml_backend_dev_t devs[] = {cpu_dev, cuda_dev};
    ggml_backend_dev_t meta_dev = ggml_backend_meta_device(devs, 2, get_split_state, NULL);
    printf("Meta device created: %s\n", ggml_backend_dev_name(meta_dev));

    ggml_backend_t meta_backend = ggml_backend_dev_init(meta_dev, NULL);
    printf("Backend created from meta device\n");

    struct ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    struct ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_set_name(a, "a");
    ggml_set_name(b, "b");

    struct ggml_tensor * result = ggml_add(ctx, a, b);
    ggml_set_name(result, "result");

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, result);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, meta_backend);
    printf("Tensors allocated on meta backend\n");

    float a_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b_data[] = {10.0f, 20.0f, 30.0f, 40.0f};
    ggml_backend_tensor_set(a, a_data, 0, sizeof(a_data));
    ggml_backend_tensor_set(b, b_data, 0, sizeof(b_data));

    ggml_backend_graph_compute(meta_backend, gf);

    float result_data[4];
    ggml_backend_tensor_get(result, result_data, 0, sizeof(result_data));

    printf("\nOperation result:\n");
    printf("a = [%.1f, %.1f, %.1f, %.1f]\n", a_data[0], a_data[1], a_data[2], a_data[3]);
    printf("b = [%.1f, %.1f, %.1f, %.1f]\n", b_data[0], b_data[1], b_data[2], b_data[3]);
    printf("result = [%.1f, %.1f, %.1f, %.1f]\n", result_data[0], result_data[1], result_data[2], result_data[3]);

    ggml_backend_buffer_free(buffer);
    ggml_backend_free(meta_backend);
    ggml_free(ctx);
    return 0;
}
