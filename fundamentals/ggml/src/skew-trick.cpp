#include <stdio.h>
#include <string.h>
#include <cmath>

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

int main(int argc, char **argv) {
    printf("GGML skew \"trick\"\n\n");

    struct ggml_init_params params = {
        .mem_size   = 16*1024*1024,
        .mem_buffer = NULL,
    };

    struct ggml_context* ctx = ggml_init(params);

    constexpr int chunk       = 2; // 2 queries per block
    constexpr int window_size = 3; // allowed window size
    constexpr int n_kv_chunk  = 4; // required Value tensor width alignment

    printf("chunk size:  %d\n", chunk);
    printf("window_size: %d\n", window_size);
    printf("n_kv_chunk:  %d\n", n_kv_chunk);
    printf("\n");

    struct ggml_tensor * content_scores = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, n_kv_chunk, chunk, 1, 1);
    
    // 99.0 represents processing waste out-of-bounds of our sliding window.
    float raw_scores[chunk][n_kv_chunk] = {
        { 10.0f, 11.0f, 12.0f, 99.0f },  // Row 0 (Query 0): 3 valid, 1 waste
        { 99.0f, 20.0f, 21.0f, 22.0f }   // Row 1 (Query 1): 1 waste, 3 valid
    };
    memcpy(content_scores->data, raw_scores, sizeof(raw_scores));

    print_tensor(content_scores,       "original tensor");
    printf("nb[0]: %ld\n", content_scores->nb[0]);
    printf("nb[1]: %ld\n", content_scores->nb[1]);
    printf("size in bytes: %ld\n", ggml_nbytes(content_scores));    
    printf("\n");

    // At this point the memory will look like this:
    // { 10.0f  11.0f  12.0f  99.0f 99.0f  20.0f  21.0f  22.0f }
    // |-------|------|------|----→|←-----|------|------|-----→|
    // 0       4      8     12    16     20     24     28     32
    // [      row 0               ][         row 1             ]
    //
    // ne[0] = 4
    //
    // nb[0] = 4
    // nb[1] = 16
    //
    // F32 and 8 elements, 4*8 = 32

    // Stride shift skew: 
    // This create a new tensor with shape of [3, 2, 1, 1], so one dimension
    // less (x/feature dimension) than the original.
    // It also sets the stride for ne1 (the stride for x does not change as that is
    // related to the type of elements that the tensor stores. So instead of advancing
    // 16 bytes to get to the next row we are advancing:
    // (chunk + window_size) * content_scores->nb[0]
    // (2 + 3) * 4 = 20
    //
    printf("stride for content_scores_clean: %ld\n", (chunk + window_size) * content_scores->nb[0]);
    struct ggml_tensor * content_scores_clean = ggml_view_4d(ctx, content_scores,
        window_size, chunk, 1, 1,
        (size_t)(chunk + window_size) * content_scores->nb[0],
        content_scores->nb[2],  // unchanged
        content_scores->nb[3],  // unchanged
        0);                     // offset
                                //
    // At this point the memory will look like this:
    // { 10.0f  11.0f  12.0f  99.0f 99.0f  20.0f  21.0f  22.0f }
    // |-------|------|------|----→|←-----|------|------|-----→|
    // 0       4      8     12    16     20     24     28     32
    // [      row 0          ]            [   row 1            ]
    //
    // ne[0] = 3
    //
    // nb[0] = 4
    // nb[1] = 20

    print_tensor(content_scores_clean,       "content_scores_clean");
    printf("nb[0]: %ld\n", content_scores_clean->nb[0]);
    printf("nb[1]: %ld\n", content_scores_clean->nb[1]);
    printf("size in bytes: %ld\n", ggml_nbytes(content_scores_clean));    
    printf("\n");
    
    // When the graph is executed later this will cause the memory to be
    // contigous in memory so we will actually physically have:
    // { 10.0f  11.0f  12.0f 20.0f  21.0f  22.0f }
    content_scores_clean = ggml_cont(ctx, content_scores_clean);

    // The above was the scew part (trick)
    //
    // The next part if what undoes it.

    // Reverse stride shift preperation: right pad the x dimension with 2
    struct ggml_tensor * probs_padded = ggml_pad_ext(ctx, content_scores_clean, 0, chunk, 0, 0, 0, 0, 0, 0);
    // { 10.0f  11.0f  12.0f 0.0f 0.0f 20.0f  21.0f  22.0f 0.0f 0.0f }
    //                        ↑    ↑                        ↑    ↑    
    // shape: [5, 2, 1, 1]

    // reverse stride shift which will use the tensor that we prepared above
    // by padding. Here we create a new tensor of shape [4, 2, 1, 1].
    // nb[1] = 4 * 4 = 16
    struct ggml_tensor * probs_unskewed = ggml_view_4d(ctx, probs_padded,
        n_kv_chunk, chunk, 1, 1,
        (size_t)n_kv_chunk * probs_padded->nb[0], // 4 * sizeof(float)
        probs_padded->nb[2],
        probs_padded->nb[3],
        0);
    // { 10.0f  11.0f  12.0f 0.0f 0.0f 20.0f  21.0f  22.0f 0.0f 0.0f }
    //   [                      ] [                      ]
    
    // When the graph is executed later this will cause the memory to be
    // contigous in memory so we will actually physically have:
    // { 10.0f  11.0f  12.0f 0.0f 0.0f 20.0f  21.0f  22.0f }
    probs_unskewed = ggml_cont(ctx, probs_unskewed);

    struct ggml_cgraph * c_graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(c_graph, probs_unskewed);
    enum ggml_status st = ggml_graph_compute_with_ctx(ctx, c_graph, 1);

    if (st != GGML_STATUS_SUCCESS) {
        printf("Graph computation failed\n");
        return 1;
    }

    print_tensor(content_scores_clean, "content_scores_clean");
    print_tensor(probs_padded,         "probs_padded");
    print_tensor(probs_unskewed,       "probs_unskewed");

    ggml_free(ctx);
    return 0;
}
