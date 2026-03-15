#include <stdio.h>
#include <string.h>
#include <math.h>
#include "ggml.h"
#include "ggml-cpu.h"

// T = 3 (Number of audio frames)
// P = 2*T - 1 = 5 (Length of the relative position ruler)

static void print_tensor_2d(struct ggml_tensor * t, const char * label) {
    printf("%s [ne[0]=%ld ne[1]=%ld]\n", label, t->ne[0], t->ne[1]);
    for (int y = 0; y < (int)t->ne[1]; y++) {
        printf("  Row %d: ", y);
        for (int x = 0; x < (int)t->ne[0]; x++) {
            printf("%6.1f", ggml_get_f32_nd(t, x, y, 0, 0));
        }
        printf("\n");
    }
    printf("\n");
}

int main(void) {
    struct ggml_init_params params = { .mem_size = 16 * 1024 * 1024, .no_alloc = false };
    struct ggml_context * ctx = ggml_init(params);

    const int T=3;
    const int P=5;

    // Each row represents a Frame (0, 1, 2)
    // Each column represents a Distance signature (-2, -1, 0, +1, +2)
    struct ggml_tensor * rel_pos_scores = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, P, T);
    
    // Use position markers: row * 10 + col to track where values come from
    // This makes it easy to see the relative shift pattern!
    float raw_data[T][P] = {
        { -2.0, -1.0, 0.0, 1.0, 2.0 },  // Frame 0: positions 0-4
        { -2.0, -1.0, 0.0, 1.0, 2.0 }, // Frame 1: positions 10-14
        { -2.0, -1.0, 0.0, 1.0, 2.0 }, // Frame 2: positions 20-24
    };

    memcpy(rel_pos_scores->data, raw_data, sizeof(raw_data));
    print_tensor_2d(rel_pos_scores, "1. Initial Rect (P x T)");

    // Swap dimensions: {P, T} -> {T, P} = {3, 5}
    struct ggml_tensor * transposed = ggml_cont(ctx, ggml_permute(ctx, rel_pos_scores, 1, 0, 2, 3));

    // Compute transpose to see it
    struct ggml_cgraph * gf_tmp = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf_tmp, transposed);
    ggml_graph_compute_with_ctx(ctx, gf_tmp, 1);
    print_tensor_2d(transposed, "1.5. After Transpose (T x P)");

    struct ggml_tensor * pad = ggml_pad(ctx, transposed, 1, 0, 0, 0); // now width 6
    struct ggml_tensor * roll = ggml_roll(ctx, pad, 1, 0, 0, 0); // shift right by 1

    // Reference: reshape_3d(ctx, tensor, n_pos, n_frame+1, n_head)
    // From {n_frame+1, n_pos, n_head} to {n_pos, n_frame+1, n_head}
    // In 2D: from {T+1, P} to {P, T+1} = {5, 4}
    struct ggml_tensor * reshape = ggml_reshape_2d(ctx, roll, P, T + 1);

    // Reference: view_3d(ctx, tensor, n_frame, n_frame, n_head, nb[1], nb[2], nb[0] * n_pos)
    struct ggml_tensor * view = ggml_view_2d(ctx, reshape, T, T,
                                             reshape->nb[1],
                                             reshape->nb[0] * P);

    // Debug: compute intermediates
    struct ggml_cgraph * gf_debug = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf_debug, pad);
    ggml_build_forward_expand(gf_debug, roll);
    ggml_build_forward_expand(gf_debug, reshape);
    ggml_graph_compute_with_ctx(ctx, gf_debug, 1);

    print_tensor_2d(pad,     "2. pad");
    print_tensor_2d(roll,    "3. roll");
    print_tensor_2d(reshape, "4. reshape");

    // Print flat memory of reshape to understand the layout
    printf("\n4.1. Flat memory of reshape (20 elements):\n");
    float *reshape_data = (float*)reshape->data;
    for (int i = 0; i < 20; i++) {
        printf("%6.1f", reshape_data[i]);
        if ((i + 1) % 5 == 0) printf(" |");
    }
    printf("\n\n");

    // Print view BEFORE cont to see what we're extracting
    printf("4.5. view (before cont) - nb[0]=%zu nb[1]=%zu offset=%zu\n",
           view->nb[0], view->nb[1], (size_t)(((char*)view->data) - ((char*)reshape->data)));
    printf("     Expected to read: [0,1,2], [-1,0,1], [-2,-1,0]\n");
    print_tensor_2d(view, "     view actual content");

    // 5. CONT to materialize
    // Reference: cont_3d(ctx, tensor, n_frame, n_frame, n_head)
    struct ggml_tensor * final_view = ggml_cont_2d(ctx, view, T, T);

    // Compute
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, final_view);
    ggml_graph_compute_with_ctx(ctx, gf, 1);

    print_tensor_2d(final_view, "5. Final Result (T x T)");

    ggml_free(ctx);
    return 0;
}
