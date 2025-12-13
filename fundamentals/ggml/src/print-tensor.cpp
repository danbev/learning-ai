#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-impl.h"
#include "ggml-cpu.h"
#include <vector>

static float ggml_get_float_value(const uint8_t * data, ggml_type type,
        const size_t * nb, size_t i0, size_t i1, size_t i2, size_t i3) {
    size_t i = i3 * nb[3] + i2 * nb[2] + i1 * nb[1] + i0 * nb[0];
    float v;
    switch (type) {
        case GGML_TYPE_F16:
            return ggml_fp16_to_fp32(*(const ggml_fp16_t *) &data[i]);
        case GGML_TYPE_F32:
            return *(const float *) &data[i];
        case GGML_TYPE_I64:
            return (float) *(const int64_t *) &data[i];
        case GGML_TYPE_I32:
            return (float) *(const int32_t *) &data[i];
        case GGML_TYPE_I16:
            return (float) *(const int16_t *) &data[i];
        case GGML_TYPE_I8:
            return (float) *(const int8_t *) &data[i];
        case GGML_TYPE_BF16:
            return ggml_compute_bf16_to_fp32(*(const ggml_bf16_t *) &data[i]);
        default:
            GGML_ABORT("fatal error");
    }
}

static void print_tensor_ms(ggml_tensor * t) {
    if (t != nullptr) {
        printf("Tensor '%s', type: %s\n", t->name, ggml_type_name(t->type));
        printf("ne = [%lld %lld %lld %lld]\n", (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3]);

        const int64_t d_model = t->ne[0];
        const int64_t total_elements = ggml_nelements(t);
        auto n_bytes = ggml_nbytes(t);

        std::vector<float> data(n_bytes);

        if (t->data == nullptr) {
             printf("Error: t->data is null\n");
             return;
        }
        memcpy(data.data(), t->data, n_bytes);

        float tmp = 0.0;
        for (int64_t i3 = 0; i3 < t->ne[3]; i3++) {
            for (int64_t i2 = 0; i2 < t->ne[2]; i2++) {
                for (int64_t i1 = 0; i1 < t->ne[1]; i1++) {
                    for (int64_t i0 = 0; i0 < t->ne[0]; i0++) {
                        uint8_t * d = (uint8_t *) data.data();
                        const float v = ggml_get_float_value(d, t->type, t->nb, i0, i1, i2, i3);
                        tmp += v * v;

                        printf("Tensor value at [%lld, %lld, %lld, %lld]: %.6f\n",
                            (long long)i3, (long long)i2, (long long)i1, (long long)i0, v);
                    }
                }
            }
        }

        double mean_sq = tmp / (double) total_elements;
        printf("%s mean_sq = %.10f\n", t->name, mean_sq);
    }
}

static void print_tensor(const char * name, ggml_backend_sched_t sched, ggml_cgraph * gf,
        size_t n_values_to_print = 0) {
    printf("values to print: %zu\n", n_values_to_print);
    struct ggml_tensor * t = ggml_graph_get_tensor(gf, name);
    if (t == nullptr) {
        printf("Tensor '%s' not found in graph.\n", name);
        return;
    }

    printf("Tensor '%s', type: %s\n", t->name, ggml_type_name(t->type));
    printf("ne = [%lld %lld %lld %lld]\n", (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3]);

    const int64_t d_model = t->ne[0];
    const int64_t total_elements = ggml_nelements(t);

    auto n_bytes = ggml_nbytes(t);
    // The tensor data type can be different so we read raw bytes.
    std::vector<uint8_t> data_bytes(n_bytes);

    ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, t);

    ggml_backend_tensor_get_async(backend, t, data_bytes.data(), 0, n_bytes);
    ggml_backend_sched_synchronize(sched);

    float tmp = 0.0;
    uint8_t * d = data_bytes.data();

    size_t values_count = 0;
    for (int64_t i3 = 0; i3 < t->ne[3]; i3++) {
        for (int64_t i2 = 0; i2 < t->ne[2]; i2++) {
            for (int64_t i1 = 0; i1 < t->ne[1]; i1++) {
                for (int64_t i0 = 0; i0 < t->ne[0]; i0++) {
                    const float v = ggml_get_float_value(d, t->type, t->nb, i0, i1, i2, i3);
                    tmp += v * v;

                    if (values_count++ < n_values_to_print) {
                        printf("Tensor value at [%lld, %lld, %lld, %lld]: %.6f\n",
                            (long long)i3, (long long)i2, (long long)i1, (long long)i0, v);
                    }
                }
            }
        }
    }

    double mean_sq = tmp / (double) total_elements;
    printf("%s mean_sq = %.10f\n", t->name, mean_sq);
}

int main() {

    {
        struct ggml_init_params params = {
            .mem_size   = 4*1024*1024,
            .mem_buffer = NULL,
        };
        struct ggml_context* ctx = ggml_init(params);

        struct ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5);
        float data[5] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        ggml_set_name(t, "tensor_1d");
        for (int i = 0; i < 5; ++i) {
            ggml_set_f32_1d(t, i, data[i]);
        }
        print_tensor_ms(t);
        ggml_free(ctx);
    }

    {
        struct ggml_init_params params = {
            .mem_size   = 4*1024*1024,
            .mem_buffer = NULL,
            .no_alloc = true,
        };
        struct ggml_context * ctx = ggml_init(params);

        struct ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 3);
        ggml_set_name(t, "tensor_2d");

        struct ggml_cgraph* c_graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(c_graph, t);

        ggml_backend_t backend = ggml_backend_init_by_name("CPU", NULL);
        ggml_backend_buffer_t b = ggml_backend_alloc_ctx_tensors(ctx, backend);

        ggml_backend_t backends[] = {backend};
        ggml_backend_buffer_type_t buffer_types[] = {
            ggml_backend_get_default_buffer_type(backend),
        };

        size_t graph_work_size = ggml_graph_size(c_graph);
        size_t graph_nodes = ggml_graph_n_nodes(c_graph);
        ggml_backend_sched_t sched = ggml_backend_sched_new(backends, buffer_types, 1, graph_nodes, graph_work_size, false);

        if (!ggml_backend_sched_alloc_graph(sched, c_graph)) {
            fprintf(stderr, "Failed to allocate graph\n");
            ggml_backend_sched_free(sched);
            ggml_backend_free(backend);
            return 1;
        }

        // We don't have any operations but just for completeness
        ggml_backend_sched_graph_compute(sched, c_graph);

        float data[6] = {6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
        ggml_backend_tensor_set(t, data, 0, ggml_nbytes(t));

        print_tensor(t->name, sched, c_graph, 2);

        ggml_backend_buffer_free(b);
        ggml_backend_free(backend);
        ggml_free(ctx);
    }

    return 0;
}
