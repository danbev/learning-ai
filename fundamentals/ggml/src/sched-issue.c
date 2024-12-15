#include <stdio.h>
#include <string.h>
#include <assert.h>

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

/*
 * Run with env GGML_SCHED_DEBUG=2 to see debug output:
 * $ GGML_SCHED_DEBUG=2 ./sched
 *
 * This examples tries to simulate a real-world scenario where one graph is
 * built and reserved in the scheduler for a language model, and then later
 * the same scheduler is used to build a different graph for a vision model.
 *
 * In llama.cpp and the new Vision API the language model it built and reserved
 * and also executed (as part of the building of the context). And the scheduler
 * is then the same for the vision model.
 */
int main(int argc, char **argv) {
    printf("GGML ggml_backend_sched issue example\n\n");

    struct ggml_init_params params = {
        .mem_size   = 16*1024*1024,
        .mem_buffer = NULL,
        .no_alloc   = true,
    };
    struct ggml_context* ctx = ggml_init(params);

    // Tensors for the first graph (simulating the language model):
    struct ggml_tensor* l_a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_input(l_a);
    ggml_set_name(l_a, "l_a");
    float l_a_data[] = {2};

    struct ggml_tensor* l_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_input(l_b);
    ggml_set_name(l_b, "l_b");
    float l_b_data[] = {3};

    struct ggml_tensor* l_r = ggml_mul(ctx, l_a, l_b);
    ggml_set_name(l_r, "l_r");

    struct ggml_cgraph* l_g = ggml_new_graph(ctx);
    ggml_build_forward_expand(l_g, l_r);

    // Create a scheduler.
    ggml_backend_t cpu_backend = ggml_backend_init_by_name("CPU", NULL);

    auto threadpool_params = ggml_threadpool_params_default(1);
    auto threadpool = ggml_threadpool_new(&threadpool_params);
    auto* reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(cpu_backend));
    auto* set_threadpool_fn = (decltype(ggml_backend_cpu_set_threadpool) *) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_threadpool");
    set_threadpool_fn(cpu_backend, threadpool);

    auto ggml_backend_set_n_threads_fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
    if (ggml_backend_set_n_threads_fn) {
        ggml_backend_set_n_threads_fn(cpu_backend, 1);
    }

    ggml_backend_t backends[] = {cpu_backend};
    ggml_backend_buffer_type_t buffer_types[] = {
        ggml_backend_get_default_buffer_type(cpu_backend)
    };
    size_t graph_size = ggml_graph_size(l_g);
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, buffer_types, 1, graph_size, false);

    if(!ggml_backend_sched_reserve(sched, l_g)) {
        fprintf(stderr, "Failed to reserve graph\n");
        ggml_backend_sched_free(sched);
        ggml_backend_free(cpu_backend);
        return 1;
    }
    
    // Allocate the graph in the scheduler.
    if (!ggml_backend_sched_alloc_graph(sched, l_g)) {
        fprintf(stderr, "Failed to allocate graph\n");
        ggml_backend_sched_free(sched);
        ggml_backend_free(cpu_backend);
        return 1;
    }

    // Tensors for the second graph, simulating the vision model:
    struct ggml_tensor* v_a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_input(v_a);
    ggml_set_name(v_a, "v_a");
    float v_a_data[] = {4};

    struct ggml_tensor* v_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_input(v_b);
    ggml_set_name(v_b, "v_b");
    float v_b_data[] = {5};

    struct ggml_tensor* v_c = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_input(v_c);
    ggml_set_name(v_c, "v_c");
    float v_c_data[] = {3};

    struct ggml_tensor* v_ab = ggml_mul(ctx, v_a, v_b);
    ggml_set_name(v_ab, "v_ab");

    struct ggml_tensor* v_r = ggml_mul(ctx, v_c, v_ab);
    ggml_set_name(v_r, "v_r");

    // Create the graph for the second model (vision model).
    struct ggml_cgraph* v_g = ggml_new_graph(ctx);
    ggml_build_forward_expand(v_g, v_r);

    // Allocate the graph in the scheduler for the second graph.
    if (!ggml_backend_sched_alloc_graph(sched, v_g)) {
        fprintf(stderr, "Failed to allocate graph\n");
        ggml_backend_sched_free(sched);
        ggml_backend_free(cpu_backend);
        return 1;
    }

    // Set inputs for the second graph:
    ggml_backend_tensor_set(v_a, v_a_data, 0, ggml_nbytes(v_a));
    ggml_backend_tensor_set(v_b, v_b_data, 0, ggml_nbytes(v_b));
    ggml_backend_tensor_set(v_c, v_c_data, 0, ggml_nbytes(v_c));
    printf("Vision model inputs:\n");
    printf("v_a: %.2f\n", v_a_data[0]);
    printf("v_b: %.2f\n", v_b_data[0]);
    printf("v_c: %.2f\n", v_c_data[0]);

    ggml_backend_sched_graph_compute(sched, v_g);

    float* v_r_data = (float*) v_r->data;
    printf("v_r (3 * (4 * 5)) = %.2f \n", v_r_data[0]);
    assert(v_r_data[0] == 60);
    printf("\n");
    
    // Now execute the first graph (language model):
    l_g = ggml_new_graph(ctx);
    ggml_build_forward_expand(l_g, l_r);
    
    ggml_backend_sched_reset(sched);
    if (!ggml_backend_sched_alloc_graph(sched, l_g)) {
        fprintf(stderr, "Failed to allocate graph\n");
        ggml_backend_sched_free(sched);
        ggml_backend_free(cpu_backend);
        return 1;
    }

    // Set tensor data for first graph:
    ggml_backend_tensor_set(l_a, l_a_data, 0, ggml_nbytes(l_a));
    ggml_backend_tensor_set(l_b, l_b_data, 0, ggml_nbytes(l_b));
    printf("Language model inputs:\n");
    printf("l_a: %.2f\n", l_a_data[0]);
    printf("l_b: %.2f\n", l_b_data[0]);

    ggml_backend_sched_graph_compute(sched, l_g);

    float* l_r_data = (float*) l_r->data;
    printf("l_r (2 * 3) = %.2f \n", l_r_data[0]);
    assert(l_r_data[0] == 6);

    ggml_free(ctx);
    return 0;
}
