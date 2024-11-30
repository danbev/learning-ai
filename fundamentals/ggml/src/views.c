#include <stdio.h>

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

int main(int argc, char **argv) {
    printf("GGML views examples\n");

    struct ggml_init_params params = {
        .mem_size   = 16*1024*1024,
        .mem_buffer = NULL,
        .no_alloc   = true,
    };
    struct ggml_context* ctx = ggml_init(params);


    struct ggml_tensor* x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 6);
    printf("x dimensions: %d\n", ggml_n_dims(x));
    printf("x elements: %ld\n", ggml_nelements(x));
    printf("x element size: %ld\n", ggml_element_size(x));
    printf("x type size: %ld\n", ggml_type_size(x->type));
    printf("x.ne: %ld\n", x->ne[0]);
    float data[] = {1, 2, 3, 4, 5, 6};
    printf("Setting data:\n");
    for (int i = 0; i < 6; i++) {
        printf("data[%d] = %f\n", i, data[i]);
    }

    size_t nb1 = ggml_element_size(x) * 2;
    size_t offset = 0;
    struct ggml_tensor* view = ggml_view_2d(ctx, x, 2, 3, nb1, offset);
    printf("view rows: %ld\n", ggml_nrows(view));
    printf("view elements: %ld\n", ggml_nelements(view));
    printf("view type: %s\n", ggml_type_name(view->type));
    printf("view ne[0]: %ld\n", view->ne[0]);
    printf("view ne[1]: %ld\n", view->ne[1]);

    ggml_backend_t backend = ggml_backend_init_by_name("CPU", NULL);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);

    ggml_backend_tensor_set(x, data, 0, ggml_nelements(x) * ggml_element_size(x));

    struct ggml_context * ctx_compute = NULL;
    struct ggml_init_params params_compute = {
        .mem_size   = 16*1024*1024,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };
    ctx_compute = ggml_init(params_compute);
    struct ggml_cgraph * gf = ggml_new_graph(ctx_compute);

    ggml_build_forward_expand(gf, view);

    ggml_backend_graph_compute(backend, gf);

    {
        printf("Loop over all dimensions in view:\n");
        float buf[6];
        ggml_backend_tensor_get(view, buf, 0, sizeof(buf));
        for (int row = 0; row < view->ne[1]; row++) {
            for (int col = 0; col < view->ne[0]; col++) {
                int idx = row * view->ne[0] + col;
                printf("buf[%d, %d] = %f\n", row, col, buf[idx]);
            }
        }
    }

    {
        printf("Loop over all elements in view:\n");
        float buf[6];
        ggml_backend_tensor_get(view, buf, 0, sizeof(buf));
        for (int i = 0; i < ggml_nelements(view); i++) {
            printf("buf[%d] = %f\n", i, buf[i]);
        }
    }

    {
        printf("Loop over all elements in src tensor:\n");
        float buf[6];
        ggml_backend_tensor_get(x, buf, 0, sizeof(buf));
        for (int i = 0; i < x->ne[0]; i++) {
            printf("buf[%d] = %f\n", i, buf[i]);
        }
    }

    ggml_free(ctx);

    return 0;
}
