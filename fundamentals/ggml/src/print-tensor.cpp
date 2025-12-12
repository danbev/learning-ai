#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-impl.h"
#include "ggml-cpu.h"
#include <vector>

static float ggml_get_float_value(const uint8_t * data, ggml_type type,
        const size_t * nb, size_t i0, size_t i1, size_t i2, size_t i3) {
    size_t i = i3 * nb[3] + i2 * nb[2] + i1 * nb[1] + i0 * nb[0];
    float v;
    if (type == GGML_TYPE_F16) {
        v = ggml_fp16_to_fp32(*(const ggml_fp16_t *) &data[i]);
    } else if (type == GGML_TYPE_F32) {
        v = *(const float *) &data[i];
    } else if (type == GGML_TYPE_I64) {
        v = (float) *(const int64_t *) &data[i];
    } else if (type == GGML_TYPE_I32) {
        v = (float) *(const int32_t *) &data[i];
    } else if (type == GGML_TYPE_I16) {
        v = (float) *(const int16_t *) &data[i];
    } else if (type == GGML_TYPE_I8) {
        v = (float) *(const int8_t *) &data[i];
    } else if (type == GGML_TYPE_BF16) {
        v = ggml_compute_bf16_to_fp32(*(const ggml_bf16_t *) &data[i]);
    } else {
        GGML_ABORT("fatal error");
    }
    return v;
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

int main() {
    struct ggml_init_params params = {
    .mem_size   = 16*1024*1024,
    .mem_buffer = NULL,
    };
    struct ggml_context* ctx = ggml_init(params);

    {
        struct ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5);
        float data[5] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        ggml_set_name(t, "tensor_1d");
        for (int i = 0; i < 5; ++i) {
            ggml_set_f32_1d(t, i, data[i]);
        }
        print_tensor_ms(t);
    }

    {
        struct ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 3);
        ggml_set_name(t, "tensor_2d");
        print_tensor_ms(t);
    }

    return 0;
}
