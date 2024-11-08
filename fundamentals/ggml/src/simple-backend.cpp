#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <vector>

struct simple_model {
    struct ggml_tensor* a = nullptr;
    struct ggml_tensor* b = nullptr;

    // the backend to perform the computation (CPU)
    ggml_backend_t backend = nullptr;

    // the backend buffer to store the tensors data of a and b
    ggml_backend_buffer_t buffer = nullptr;

    // the context to define the tensor information (dimensions, size, memory address)
    struct ggml_context* ctx = nullptr;
};

void load_model(simple_model& model,
                float* a,
                float* b,
                int rows_a,
                int cols_a,
                int rows_b,
                int cols_b) {
    model.backend = ggml_backend_cpu_init();

    int num_tensors = 2;

    struct ggml_init_params params {
            /*.mem_size   =*/ ggml_tensor_overhead() * num_tensors,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
    };

    // create context
    model.ctx = ggml_init(params);

    // create tensors
    model.a = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, cols_a, rows_a);
    model.b = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, cols_b, rows_b);

    // create a backend buffer (backend memory) and alloc the tensors from the context
    model.buffer = ggml_backend_alloc_ctx_tensors(model.ctx, model.backend);

    // load data from cpu memory to backend buffer
    ggml_backend_tensor_set(model.a, a, 0, ggml_nbytes(model.a));
    ggml_backend_tensor_set(model.b, b, 0, ggml_nbytes(model.b));
}

// build the compute graph to perform a matrix multiplication
struct ggml_cgraph* build_graph(const simple_model& model) {
    static size_t buf_size = ggml_tensor_overhead() *GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
    static std::vector<uint8_t> buf(buf_size);

    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf.data(),
        /*.no_alloc   =*/ true, // the tensors will be allocated later by ggml_allocr_alloc_graph()
    };

    // create a temporary context to build the graph.
    struct ggml_context* ctx = ggml_init(params);

    struct ggml_cgraph* gf = ggml_new_graph(ctx);

    // result = a*b^T   (notice the implicit transpose of b)
    struct ggml_tensor* result = ggml_mul_mat(ctx, model.a, model.b);

    // build operations nodes
    ggml_build_forward_expand(gf, result);

    // delete the temporary context used to build the graph
    ggml_free(ctx);

    return gf;
}

// compute with backend
struct ggml_tensor* compute(const simple_model& model, ggml_gallocr_t allocr) {
    // reset the allocator to free all the memory allocated during the previous inference

    struct ggml_cgraph* gf = build_graph(model);

    // allocate tensors
    ggml_gallocr_alloc_graph(allocr, gf);

    int n_threads = 1; // number of threads to perform some operations with multi-threading

    if (ggml_backend_is_cpu(model.backend)) {
        ggml_backend_cpu_set_n_threads(model.backend, n_threads);
    }

    ggml_backend_graph_compute(model.backend, gf);

    // in this case, the output tensor is the last one in the graph
    return gf->nodes[gf->n_nodes - 1];
}

int main(void) {
    ggml_time_init();

    const int rows_a = 2, cols_a = 2;

    float matrix_A[rows_a * cols_a] = {
        1, 2,
        3, 4
    };

    const int rows_b = 3, cols_b = 2;
    float matrix_B[rows_b * cols_b] = {
        5, 8,
        6, 9,
        7, 10
    };

    simple_model model;
    load_model(model, matrix_A, matrix_B, rows_a, cols_a, rows_b, cols_b);

    // calculate the temporaly memory required to compute
    ggml_gallocr_t allocr = nullptr;
    {
        allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));

        // create the worst case graph for memory usage estimation
        struct ggml_cgraph * gf = build_graph(model);
        ggml_gallocr_reserve(allocr, gf);
        size_t mem_size = ggml_gallocr_get_buffer_size(allocr, 0);

        fprintf(stderr, "%s: compute buffer size: %.4f KB\n", __func__, mem_size/1024.0);
    }

    // perform computation
    struct ggml_tensor * result = compute(model, allocr);

    // create a array to print result
    std::vector<float> out_data(ggml_nelements(result));

    // bring the data from the backend memory
    ggml_backend_tensor_get(result, out_data.data(), 0, ggml_nbytes(result));

    printf("mul mat (%d x %d) :\n[", (int) result->ne[0], (int) result->ne[1]);
    for (int j = 0; j < result->ne[0] /* rows */; j++) {
        if (j > 0) {
            printf("\n");
        }

        for (int i = 0; i < result->ne[1] /* cols */; i++) {
            printf(" %.2f", out_data[i * result->ne[0] + j]);
        }
    }
    printf(" ]\n");

    // release backend memory used for computation
    ggml_gallocr_free(allocr);

    // free memory
    ggml_free(model.ctx);

    // release backend memory and free backend
    ggml_backend_buffer_free(model.buffer);
    ggml_backend_free(model.backend);
    return 0;
}
