#include <stdio.h>
#include <string.h>

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

/*
 * Run with env GGML_SCHED_DEBUG=2 to see debug output:
 * $ GGML_SCHED_DEBUG=2 ./sched
 *
 */
int main(int argc, char **argv) {
    printf("GGML ggml_backend_sched example\n\n");

    struct ggml_init_params params = {
        .mem_size   = 16*1024*1024,
        .mem_buffer = NULL,
        .no_alloc   = true,
    };
    struct ggml_context* ctx = ggml_init(params);

    // Tensors for the first graph:
    struct ggml_tensor* a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_input(a);
    ggml_set_name(a, "a");
    float a_data[] = {2};

    struct ggml_tensor* b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_input(b);
    ggml_set_name(b, "b");
    float b_data[] = {3};

    struct ggml_tensor* mul = ggml_mul(ctx, a, b);
    ggml_set_name(mul, "mul");

    struct ggml_cgraph* g1 = ggml_new_graph(ctx);
    ggml_build_forward_expand(g1, mul);


    ggml_backend_t cpu_backend = ggml_backend_init_by_name("CPU", NULL);
    ggml_backend_t backends[] = {cpu_backend};
    ggml_backend_buffer_type_t buffer_types[] = {
        ggml_backend_get_default_buffer_type(cpu_backend)
    };
    size_t graph_size = ggml_graph_size(g1);
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, buffer_types, 1, graph_size, false, false);
    if (!sched) {
        fprintf(stderr, "Failed to create scheduler\n");
        ggml_backend_free(cpu_backend);
        return 1;
    }
    
    if (!ggml_backend_sched_alloc_graph(sched, g1)) {
        fprintf(stderr, "Failed to allocate graph\n");
        ggml_backend_sched_free(sched);
        ggml_backend_free(cpu_backend);
        return 1;
    }

    // Set tensor data for first graph:
    ggml_backend_tensor_set(a, a_data, 0, ggml_nbytes(a));
    ggml_backend_tensor_set(b, b_data, 0, ggml_nbytes(b));
    printf("a: %.2f\n", a_data[0]);
    printf("b: %.2f\n", b_data[0]);

    ggml_backend_sched_graph_compute(sched, g1);

    printf("\nresult:\n");
    float* mul_data = (float*) mul->data;
    for (int i = 0; i < ggml_nelements(mul); i++) {
        printf("%.2f ", mul_data[i]);
    }
    printf("\n");

    {
        struct ggml_tensor * tensor = ggml_graph_get_tensor(g1, "a");
        float buf[1];
        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, tensor);
        printf("Tensor type: %s\n", ggml_type_name(tensor->type));
        printf("Backend type: %s\n", ggml_backend_name(backend));
        ggml_backend_tensor_get_async(backend, tensor, buf, 0, sizeof(buf));
        ggml_backend_sched_synchronize(sched);
        printf("a: %f (notice that a was overwritten!)\n", buf[0]);
    }

    {
        struct ggml_tensor * tensor = ggml_graph_get_tensor(g1, "b");
        float buf[1];
        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, tensor);
        ggml_backend_tensor_get_async(backend, tensor, buf, 0, sizeof(buf));
        ggml_backend_sched_synchronize(sched);
        printf("b: %f\n", buf[0]);
    }


    // Tensors for the second graph:
    struct ggml_tensor* a2 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_input(a2);
    ggml_set_name(a2, "a2");
    float a2_data[] = {4};

    struct ggml_tensor* b2 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_input(b2);
    ggml_set_name(b2, "b");
    float b2_data[] = {5};

    struct ggml_tensor* c = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_input(c);
    ggml_set_name(c, "c");
    float c_data[] = {3};

    struct ggml_tensor* ab = ggml_mul(ctx, a2, b2);
    struct ggml_tensor* result = ggml_mul(ctx, c, ab);
    ggml_set_name(result, "result");

    struct ggml_cgraph* g2 = ggml_new_graph(ctx);
    ggml_build_forward_expand(g2, result);

    if (!ggml_backend_sched_alloc_graph(sched, g2)) {
        fprintf(stderr, "Failed to allocate graph 2\n");
        ggml_backend_sched_free(sched);
        ggml_backend_free(cpu_backend);
        return 1;
    }

    ggml_backend_tensor_set(a2, a2_data, 0, ggml_nbytes(a2));
    ggml_backend_tensor_set(b2, b2_data, 0, ggml_nbytes(b2));
    ggml_backend_tensor_set(c,   c_data, 0, ggml_nbytes(c));

    ggml_backend_sched_graph_compute(sched, g2);

    float* result_data = (float*) result->data;
    printf("result: %.2f \n", result_data[0]);

    {
        struct ggml_tensor * tensor = ggml_graph_get_tensor(g2, "c");
        float buf[1];
        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, tensor);
        ggml_backend_tensor_get_async(backend, tensor, buf, 0, sizeof(buf));
        ggml_backend_sched_synchronize(sched);
        printf("c: %.2f\n", buf[0]);
    }

    ggml_free(ctx);
    return 0;
}
