#include <stdio.h>
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include <cstring>

int main(int argc, char **argv) {
    printf("ggml_unravel_index example\n");
    
    struct ggml_init_params params = {
        .mem_size   = 16*1024*1024,
        .mem_buffer = NULL,
    };
    struct ggml_context* ctx = ggml_init(params);

    int32_t data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
    //                0  1  2  3  4  5  6  7  8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23
    int idx = 12;
    
    struct ggml_tensor* tensor = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, 3, 2, 2, 2);
    printf("Tensor created with shape (%d,%d,%d,%d)\n",
            (int)tensor->ne[0], (int)tensor->ne[1],(int)tensor->ne[2], (int)tensor->ne[3]);
    ggml_set_name(tensor, "4_d");
    memcpy(tensor->data, data, ggml_nbytes(tensor));
    
    int64_t i[4] = {0, 0, 0, 0};
    ggml_unravel_index(tensor, idx, &i[0], &i[1], &i[2], &i[3]);

    printf("Unraveled index of %d in tensor shape (%d,%d,%d,%d): (%ld, %ld, %ld, %ld)\n",
            idx,
            (int)tensor->ne[0], (int)tensor->ne[1],(int)tensor->ne[2], (int)tensor->ne[3],
            i[0], i[1], i[2], i[3]);

   int value = ggml_get_i32_nd(tensor, i[0], i[1], i[2], i[3]);
   printf("Value at unraveled index: %d\n", value);

    
    ggml_free(ctx);
    return 0;
}
