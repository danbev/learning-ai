#include "ggml.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>

int main(int argc, char** argv) {
    printf("GGML mem_buffer example\n");

    size_t mem_size = GGML_PAD(ggml_tensor_overhead() + sizeof(int),GGML_MEM_ALIGN);
    printf("mem_size: %d\n", (int) mem_size);
    void * mem_buffer = malloc(mem_size);

    struct ggml_tensor * a;
    {
        struct ggml_init_params params = {
            .mem_size   = mem_size,
            .mem_buffer = mem_buffer,
            .no_alloc   = false,
        };

        struct ggml_context * ctx = ggml_init(params);

        a = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        ggml_set_name(a, "a");
        int data = 18;
        memcpy(a->data, &data, ggml_nbytes(a));

        printf("tensor a data: %p\n", a->data);
        printf("tensor a data: %d\n", *(int *)a->data);

        ggml_free(ctx);
    }

    printf("tensor a data pointer: %p\n", a->data);
    printf("tensor a data: %d\n", *(int *)a->data);

    free(mem_buffer);
    return 0;
}
