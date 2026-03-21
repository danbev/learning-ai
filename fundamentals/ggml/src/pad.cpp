#include <stdio.h>
#include <string.h>

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

static void print_tensor(struct ggml_tensor * t, const char * label) {
    printf("%s [%d, %d, %d, %d]\n", label, (int)t->ne[0], (int)t->ne[1], (int)t->ne[2], (int)t->ne[3]);

    for (int b = 0; b < t->ne[3]; ++b) {
        for (int c = 0; c < t->ne[2]; ++c) {
            for (int y = 0; y < (int)t->ne[1]; y++) {
                printf("  Row %d: ", y);
                for (int x = 0; x < (int)t->ne[0]; x++) {
                    printf("%6.1f", ggml_get_f32_nd(t, x, y, 0, 0));
                }
                printf("\n");
            }
        }
    }
    printf("\n");
}

int main(int argc, char **argv) {
    printf("ggml_pad example\n");

    struct ggml_init_params params = {
    .mem_size   = 16*1024*1024,
    .mem_buffer = NULL,
    };

    struct ggml_context* ctx = ggml_init(params);

    constexpr int ne0=5;
    constexpr int ne1=3;

    struct ggml_tensor* tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1);

    float raw_data[ne1][ne0] = {
        { -2.0, -1.0, 0.0, 1.0, 2.0 },
        { -2.0, -1.0, 0.0, 1.0, 2.0 },
        { -2.0, -1.0, 0.0, 1.0, 2.0 },
    };
    memcpy(tensor->data, raw_data, sizeof(raw_data));

    // The following will add padding of one column of zeros.
    // And we can add a row of padding by specifying 1 for the second argument.
    struct ggml_tensor* result = ggml_pad(ctx, tensor, 1, 0, 0, 0);
    ggml_set_name(result, "result");

    struct ggml_cgraph* c_graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(c_graph, result);
    enum ggml_status st = ggml_graph_compute_with_ctx(ctx, c_graph, 1);

    if (st != GGML_STATUS_SUCCESS) {
        printf("could not compute graph\n");
        return 1;
    }

    print_tensor(result, "result");

    ggml_free(ctx);
    return 0;
}
