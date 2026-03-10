#include <stdio.h>
#include <string.h>
#include "ggml.h"
#include "ggml-cpu.h"

#define NX 4
#define NY 3

static void print_tensor_2d(struct ggml_tensor * t, const char * label) {
    printf("%s  [ne[0]=%ld  ne[1]=%ld]\n", label, t->ne[0], t->ne[1]);
    for (int y = 0; y < (int)t->ne[1]; y++) {
        printf("  [%d]: ", y);
        for (int x = 0; x < (int)t->ne[0]; x++) {
            printf("%6.1f", ggml_get_f32_nd(t, x, y, 0, 0));
        }
        printf("\n");
    }
    printf("\n");
}

int main(void) {
    printf("=== ggml_roll example ===\n\n");

    struct ggml_init_params params = {
        .mem_size   = 16 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, NX, NY);
    ggml_set_name(a, "a");

    float src[NY][NX] = {
        {   1,   2,   3,   4 },
        {  10,  20,  30,  40 },
        { 100, 200, 300, 400 },
    };
    memcpy(a->data, src, sizeof(src));
    print_tensor_2d(a, "input");

    struct ggml_tensor * roll_cols = ggml_roll(ctx, a, 1, 0, 0, 0);
    ggml_set_name(roll_cols, "roll_cols_+2");

    struct ggml_cgraph * cgraph = ggml_new_graph(ctx);
    ggml_build_forward_expand(cgraph, roll_cols);

    ggml_graph_compute_with_ctx(ctx, cgraph, /*n_threads=*/1);

    print_tensor_2d(roll_cols, "roll(shift0=+1, shift1=0)  -- shift cols right by 1");

    ggml_free(ctx);
    return 0;
}
