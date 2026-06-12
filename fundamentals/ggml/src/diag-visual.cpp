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
                    printf("%6.1f", ggml_get_f32_nd(t, x, y, c, b));
                }
                printf("\n");
            }
        }
    }
    printf("\n");
}


static void print_tensor_layout(struct ggml_tensor * t, const char * label) {
    printf("=== %s ===\n", label);
    printf("Shape: [%d, %d]\n", (int)t->ne[0], (int)t->ne[1]);
    for (int y = 0; y < (int)t->ne[1]; y++) {
        printf("  Row %d (Query %d): ", y, y);
        for (int x = 0; x < (int)t->ne[0]; x++) {
            float val = ggml_get_f32_nd(t, x, y, 0, 0);
            if (val == 40.0f || val == 100.0f) {
                // Highlight the waste data cells
                printf("%7.1f(WASTE)", val);
            } else {
                printf("%7.1f      ", val);
            }
        }
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char ** argv) {
    struct ggml_init_params params = {
        .mem_size   = 16*1024*1024,
        .mem_buffer = NULL,
    };
    struct ggml_context* ctx = ggml_init(params);

    constexpr int d_head      = 1; // 1 feature per frame to keep math clear
    constexpr int chunk       = 2; // 2 queries processed simultaneously 
    constexpr int n_kv_chunk  = 4; // 4 keys needed to cover the sliding window range
    constexpr int window_size = 3; // each query can only look at 3 keys maximum
    printf("chunk:       %d\n", chunk);
    printf("n_kv_chunk:  %d\n", n_kv_chunk);
    printf("window_size: %d\n", window_size);
    printf("d_head:      %d\n", d_head);
    printf("\n");

    // Define 2 Queries (Identifiable values: 10.0 and 100.0)
    struct ggml_tensor * Q = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_head, chunk);
    float q_data[chunk][d_head] = {
        {  10.0f }, // query 0 can look at keys 0, 1, 2
        { 100.0f }  // query 1 can look at keys 1, 2, 3
    };
    memcpy(Q->data, q_data, sizeof(q_data));
    print_tensor(Q, "Q");

    // Define 4 Keys (Identifiable values: 1.0, 2.0, 3.0, 4.0)
    struct ggml_tensor * K = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_head, n_kv_chunk);
    float k_data[n_kv_chunk][d_head] = {
        { 1.0f },
        { 2.0f },
        { 3.0f },
        { 4.0f }
    };
    memcpy(K->data, k_data, sizeof(k_data));
    print_tensor(K, "K");

    // We perform normal matrix multipliction, there is not concept/knowledge
    // of which keys the query can look at at this stage. We make them as WASTE
    // but this only something we do when printing. The values are still there.
    struct ggml_tensor * content_scores = ggml_mul_mat(ctx, K, Q);

    // 
    // content_scores Shape: [4, 2]:
    //                           Q_1 should not see this only (1, 2, 3)
    //                             ↓
    // { 10.0  20.0  30.0  40.0  100.0 200.0  300.0  400.0 }
    //                      ↑                          
    //                    Q_0 should
    //                    not see this only (0, 1, 2).
    //
    // Create a tensor of shape [3, 2, 1, 1]
    // ne[1] = (chunk + window_size) * content_scores->nb[0]
    //       = (     2   3         )  * 4
    //       = 20
    //
    // [    row 0       ]              [   row 1         ]
    // { 10.0  20.0  30.0  40.0  100.0 200.0  300.0  400.0 }
    // [------------------------------]
    //         20 bytes
    struct ggml_tensor* content_scores_clean = ggml_view_4d(ctx, content_scores,
        window_size, chunk, 1, 1,
        (size_t)(chunk + window_size) * content_scores->nb[0], // (2 + 3) * 4 = 20 bytes
        content_scores->nb[2],
        content_scores->nb[3],
        0);

    content_scores_clean = ggml_cont(ctx, content_scores_clean);

    struct ggml_cgraph* c_graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(c_graph, content_scores_clean);
    ggml_graph_compute_with_ctx(ctx, c_graph, 1);

    print_tensor_layout(content_scores, "1. RAW MATRIX RETURNED BY GGML_MUL_MAT");
    print_tensor_layout(content_scores_clean, "2. POST-STRIDE SHIFT VIEW (DIAGONAL STRAIGHTENED)");

    ggml_free(ctx);
    return 0;
}
