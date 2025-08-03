#include <stdio.h>
#include <string.h>

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

int main(int argc, char **argv) {
    printf("GGML conv_2d example\n\n");

    struct ggml_init_params params = {
        .mem_size   = 16*1024*1024,
        .mem_buffer = NULL,
        .no_alloc   = true,
    };
    struct ggml_context* ctx = ggml_init(params);

    struct ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 2);
    ggml_set_input(a);
    ggml_set_name(a, "a");
    float a_data[] = {1, 2, 3, 4};

    struct ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 2);
    ggml_set_input(b);
    ggml_set_name(b, "b");
    float b_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

    int stride_x = 1;
    int stride_y = 1;
    int pad_x = 0;
    int pad_y = 0;
    // Dilation ("atrous convoluation" form the French term "with holes") rate.
    // 0 means no kernel elementes will be applied.
    // 1 means no dilation.
    // 2 means one space between kernel elements.
    int dil_x = 1;
    int dil_y = 1;
    struct ggml_tensor* conv = ggml_conv_2d(ctx, a, b, stride_x, stride_y, pad_x, pad_y, dil_x, dil_y);
    ggml_set_name(conv, "conv");

    struct ggml_cgraph* c_graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(c_graph, conv);


    ggml_backend_t cpu_backend = ggml_backend_init_by_name("CPU", NULL);
    ggml_backend_t cuda_backend = ggml_backend_init_by_name("CUDA0", NULL);
    if (!cuda_backend) {
        fprintf(stderr, "Failed to initialize CUDA backend\n");
        return 1;
    }
    ggml_backend_t backends[] = {cuda_backend, cpu_backend};

    // Get buffer types
    ggml_backend_buffer_type_t buffer_types[] = {
        ggml_backend_get_default_buffer_type(cuda_backend),
        ggml_backend_get_default_buffer_type(cpu_backend)
    };

    printf("CUDA buffer type: %s\n", ggml_backend_buft_name(buffer_types[0]));
    printf("CPU buffer type: %s\n", ggml_backend_buft_name(buffer_types[1]));

    size_t graph_work_size = ggml_graph_size(c_graph);
    size_t graph_nodes = ggml_graph_n_nodes(c_graph);
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, buffer_types, 2, graph_nodes, graph_work_size);
    if (!sched) {
        fprintf(stderr, "Failed to create scheduler\n");
        ggml_backend_free(cpu_backend);
        ggml_backend_free(cuda_backend);
        return 1;
    }
    
    if (!ggml_backend_sched_alloc_graph(sched, c_graph)) {
        fprintf(stderr, "Failed to allocate graph\n");
        ggml_backend_sched_free(sched);
        ggml_backend_free(cpu_backend);
        ggml_backend_free(cuda_backend);
        return 1;
    }

    ggml_backend_tensor_set(a, a_data, 0, ggml_nbytes(a));
    ggml_backend_tensor_set(b, b_data, 0, ggml_nbytes(b));

    printf("a (convolution kernel)\n");
    for (int y = 0; y < a->ne[1]; y++) {
        for (int x = 0; x < a->ne[0]; x++) {
            printf("%.2f ", *(float *) ((char *) a->data + y * a->nb[1] + x * a->nb[0]));
        }
        printf("\n");
    }
    printf("\n");

    printf("b (data):\n");
    for (int y = 0; y < b->ne[1]; y++) {
        for (int x = 0; x < b->ne[0]; x++) {
            printf("%.2f ", *(float *) ((char *) b->data + y * b->nb[1] + x * b->nb[0]));
        }
        printf("\n");
    }
    printf("\n");

    ggml_backend_sched_graph_compute(sched, c_graph);

    // Print results
    printf("\nresult:\n");
    float* conv_data = (float*)conv->data;
    for (int i = 0; i < ggml_nelements(conv); i++) {
        printf("%.2f ", conv_data[i]);
    }
    printf("\n");

    {
        struct ggml_tensor * tensor = ggml_graph_get_tensor(c_graph, "a");
        float buf[4];
        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, tensor);
        printf("Tensor type: %s\n", ggml_type_name(tensor->type));
        printf("Backend type: %s\n", ggml_backend_name(backend));
        ggml_backend_tensor_get_async(backend, tensor, buf, 0, sizeof(buf));
        ggml_backend_sched_synchronize(sched);
        for (int i = 0; i < 4; i++) {
            printf("a[%d] = %f\n", i, buf[i]);
        }
    }

    ggml_free(ctx);
    return 0;
}
