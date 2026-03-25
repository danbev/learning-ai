#include <stdio.h>
#include <string.h>
#include <vector>
#include <cmath>
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
    struct ggml_init_params params = { .mem_size = 16 * 1024 * 1024, .no_alloc = false };
    struct ggml_context * ctx = ggml_init(params);

    const int T = 3;             // n_time (rows)
    const int P = 2 * T - 1;     // pos_len (ruler width)
    const int H = 1;             // n_head (to check 3D strides)

    struct ggml_tensor * rel_pos_scores = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, P, T, H);
    ggml_set_name(rel_pos_scores, "rel_pos_scores");
    std::vector<float> data(ggml_nelements(rel_pos_scores));
    int center = P / 2; // Index 2 for P=5
    for (int h = 0; h < H; ++h) {
        for (int t = 0; t < T; ++t) {
            for (int p = 0; p < P; ++p) {
                float val = (float)(p - center); 
                val += (h + 1) * 100.0f; 
                data[h*T*P + t*P + p] = val;
            }
        }
    }
    memcpy(rel_pos_scores->data, data.data(), data.size() * sizeof(float));

    struct ggml_tensor * padded = ggml_pad(ctx, rel_pos_scores, 1, 0, 0, 0);
    ggml_set_name(padded, "padded");

    struct ggml_tensor * rolled = ggml_roll(ctx, padded, 1, 0, 0, 0);
    ggml_set_name(rolled, "rolled");

    struct ggml_tensor * reshaped = ggml_reshape_3d(ctx, rolled, T, P + 1, H);
    ggml_set_name(reshaped, "reshaped");

    size_t offset = (center+1) * sizeof(float); 

    struct ggml_tensor * sliced = ggml_view_3d(ctx, reshaped, 
        T, T, H,
        (P) * sizeof(float),  // stride for rows
        reshaped->nb[2],      // stride for heads
        offset
    );
    ggml_set_name(sliced, "sliced");

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, sliced);
    ggml_graph_compute_with_ctx(ctx, gf, 1);

    print_tensor(rel_pos_scores);
    print_tensor(padded);
    print_tensor(rolled);
    print_tensor(reshaped);
    print_tensor(sliced);

    for (int h = 0; h < H; ++h) {
        printf("--- Head %d ---\n", h);
        for (int y = 0; y < T; ++y) {
            for (int x = 0; x < T; ++x) {
                float val = ggml_get_f32_nd(sliced, x, y, h, 0);
                printf("%7.1f ", val);
            }
            printf("\n");
        }
    }

    ggml_free(ctx);
    return 0;
}
