#include <stdio.h>
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include <cstring>

void print_2d_tensor(struct ggml_tensor* tensor) {
    int cols = tensor->ne[0];
    int rows = tensor->ne[1];
    printf("Tensor %s content:\n", ggml_get_name(tensor));
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            char* base_ptr = (char*) tensor->data;
            char* element_ptr = base_ptr + (row * tensor->nb[1]) + (col * tensor->nb[0]);
            int32_t* value_ptr = (int32_t*)element_ptr;
            printf("%d ", *value_ptr);
        }
        printf("\n");
    }
}

void print_raw_memory(struct ggml_tensor* tensor) {
    printf("Raw memory contents for tensor %s: ", ggml_get_name(tensor));
    int32_t* data = (int32_t*)tensor->data;
    size_t num_elements = ggml_nelements(tensor);
    for (size_t i = 0; i < num_elements; i++) {
        printf("%d ", data[i]);
    }
    printf("\n");
}

int main(int argc, char **argv) {
    printf("ggml_is_contiguous example\n");
    
    struct ggml_init_params params = {
        .mem_size   = 16*1024*1024,
        .mem_buffer = NULL,
    };
    struct ggml_context* ctx = ggml_init(params);

    int32_t data[] = {1, 2, 3, 4, 5, 6};
    //                0  1  2  3  4  5
    
    struct ggml_tensor* tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 3, 2);
    printf("Tensor created with shape (%d,%d,%d,%d)\n",
            (int)tensor->ne[0], (int)tensor->ne[1],(int)tensor->ne[2], (int)tensor->ne[3]);
    ggml_set_name(tensor, "orginal");
    memcpy(tensor->data, data, ggml_nbytes(tensor));
    print_2d_tensor(tensor);
    print_raw_memory(tensor);
    
    printf("tensor is contiguous: %s\n", ggml_is_contiguous(tensor) ? "true" : "false ");

    struct ggml_tensor* transposed = ggml_transpose(ctx, tensor);
    printf("Transposed tensor shape (%d,%d,%d,%d)\n",
            (int)transposed->ne[0], (int)transposed->ne[1],(int)transposed->ne[2], (int)transposed->ne[3]);
    printf("transposed tensor is contiguous: %s\n", ggml_is_contiguous(transposed) ? "true" : "false ");
    print_2d_tensor(transposed);
    print_raw_memory(transposed);
    
    ggml_free(ctx);
    return 0;
}
