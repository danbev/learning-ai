#include <stdio.h>
#include <string.h>
#include <math.h>
#include "ggml.h"
#include "ggml-cpu.h"

static void print_tensor(struct ggml_tensor * t) {
    printf("%s [%d, %d, %d, %d]\n", t->name, (int)t->ne[0], (int)t->ne[1], (int)t->ne[2], (int)t->ne[3]);

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

int main(void) {
    struct ggml_init_params params = {
        .mem_size = 16 * 1024 * 1024,
        .no_alloc = false
    };
    struct ggml_context * ctx = ggml_init(params);

    constexpr int T=3;

    struct ggml_tensor * rel_pos_scores = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T, T);
    ggml_set_name(rel_pos_scores, "rel_pos_scores");
    
    // Use position markers: row * 10 + col to track where values come from
    // This makes it easy to see the relative shift pattern!
    float raw_data[T][T] = {
        { 1, 2, 3 },
        { 1, 2, 3 },
        { 1, 2, 3 }
    };
    memcpy(rel_pos_scores->data, raw_data, sizeof(raw_data));
    print_tensor(rel_pos_scores);

    struct ggml_tensor * padded = ggml_pad(ctx, rel_pos_scores, 1, 0, 0, 0);
    ggml_set_name(padded, "padded");

    struct ggml_tensor * rolled = ggml_roll(ctx, padded, 1, 0, 0, 0);
    ggml_set_name(rolled, "rolled");

    struct ggml_tensor * reshaped = ggml_reshape_2d(ctx, rolled, T, T + 1);
    ggml_set_name(reshaped, "reshaped");

    struct ggml_tensor * sliced = ggml_view_2d(ctx, reshaped, T, T,
                                              reshaped->nb[1],
                                              0);
    ggml_set_name(sliced, "sliced");

    struct ggml_cgraph * gf_debug = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf_debug, padded);
    ggml_build_forward_expand(gf_debug, rolled);
    ggml_build_forward_expand(gf_debug, reshaped);
    ggml_build_forward_expand(gf_debug, sliced);
    ggml_graph_compute_with_ctx(ctx, gf_debug, 1);

    print_tensor(padded);
    print_tensor(rolled);
    print_tensor(reshaped);
    printf("[");
    float * data = (float *)reshaped->data;
    for (int i=0; i < ggml_nelements(reshaped); ++i) {
        printf("%f, ", data[i]);
    }
    printf("]\n");
    print_tensor(sliced);

    ggml_free(ctx);
    return 0;
}
