#include <stdio.h>

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

int main(int argc, char **argv) {
    printf("GGML tensor slice example\n");

    struct ggml_init_params params = {
        .mem_size   = 16*1024*1024,
        .mem_buffer = NULL,
    };
    struct ggml_context* ctx = ggml_init(params);

    struct ggml_tensor* x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1024, 1025, 1);
    printf("x ne[0]: %ld\n", x->ne[0]);
    printf("x ne[1]: %ld\n", x->ne[1]);
    printf("x ne[2]: %ld\n", x->ne[2]);

    // Calculate the offset to skip the first row
    size_t offset = x->nb[1];  // Skip one row worth of elements
    printf("offset: %ld\n", offset);

    /*
     * z_1
     *    0 [0                     1023]
     *      ...
     *      ...
     *      ...
     * 1024 [0                     1023]
     *
     * And we want to slice it to similar to the following pytorch code:
     * x = [:, 1:, :]
     * This means include everything from the first dimension (;), and
     * the start from the second element (1:) in the second dimension, and
     * again include everything in the third dimension (:).
     *
     * z_1
     *    1 [0                     1023]
     *      ...
     *      ...
     *      ...
     * 1023 [0                     1023]
     */

    struct ggml_tensor* sliced = ggml_view_3d(ctx, x,
                                             1024,      // width (features)
                                             1024,      // height (sequence length after removing first token)
                                             1,         // depth (batch size)
                                             x->nb[0],  // row stride (unchanged)
                                             x->nb[1],  // slice stride (unchanged)
                                             offset);   // offset to skip first row

    printf("sliced tensor type: %s\n", ggml_type_name(sliced->type));
    printf("sliced ne[0]: %ld\n", sliced->ne[0]);
    printf("sliced ne[1]: %ld\n", sliced->ne[1]);
    printf("sliced ne[2]: %ld\n", sliced->ne[2]);

    ggml_free(ctx);
    return 0;
}
