#include <stdio.h>

#include "ggml.h"
#include "ggml-backend.h"

/*
 * Run with env GGML_SCHED_DEBUG=2 to see debug output:
 * $ GGML_SCHED_DEBUG=2 ./bin/sched-split
 *
 */
int main(int argc, char **argv) {
    printf("GGML sched split example\n\n");

    struct ggml_init_params params = {
        .mem_size   = 16*1024*1024,
        .mem_buffer = NULL,
        .no_alloc   = true,
    };
    struct ggml_context* ctx = ggml_init(params);

    struct ggml_tensor* a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_input(a);
    ggml_set_name(a, "a");

    struct ggml_tensor* b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_input(b);
    ggml_set_name(b, "b");

    struct ggml_tensor* mul = ggml_mul(ctx, a, b);
    ggml_set_name(mul, "mul");

    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, mul);

    ggml_backend_t cpu_backend = ggml_backend_init_by_name("CPU", NULL);
    ggml_backend_t backends[] = {cpu_backend};
    ggml_backend_buffer_type_t buffer_types[] = {
        ggml_backend_get_default_buffer_type(cpu_backend)
    };

    size_t graph_size = ggml_graph_size(gf);
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, buffer_types, 1, graph_size, false, false);
    if (!sched) {
        fprintf(stderr, "Failed to create scheduler\n");
        ggml_backend_free(cpu_backend);
        return 1;
    }
    
    if (!ggml_backend_sched_alloc_graph(sched, gf)) {
        fprintf(stderr, "Failed to allocate graph\n");
        ggml_backend_sched_free(sched);
        ggml_backend_free(cpu_backend);
        return 1;
    }

    float a_data[] = {2};
    float b_data[] = {3};
    ggml_backend_tensor_set(a, a_data, 0, ggml_nbytes(a));
    ggml_backend_tensor_set(b, b_data, 0, ggml_nbytes(b));
    printf("a: %.2f\n", a_data[0]);
    printf("b: %.2f\n", b_data[0]);

    auto status = ggml_backend_sched_graph_compute_async(sched, gf);
    if (status != GGML_STATUS_SUCCESS) {
        printf("ggml_backend_sched_graph_compute_async failed with error %d\n", status);
        return 1;
    }

    {
        struct ggml_tensor * tensor = mul;
        float buf[1];
        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, tensor);
        printf("Tensor type: %s\n", ggml_type_name(tensor->type));
        printf("Backend type: %s\n", ggml_backend_name(backend));
        ggml_backend_tensor_get_async(backend, tensor, buf, 0, sizeof(buf));
        ggml_backend_sched_synchronize(sched);
        printf("result: %1.f\n", buf[0]);
    }

    ggml_free(ctx);
    return 0;
}
