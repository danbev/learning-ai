#include <stdio.h>
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

void print_tensor_2d(struct ggml_tensor* tensor, const char* name) {
    printf("%s tensor (%d x %d):\n", name, (int)tensor->ne[0], (int)tensor->ne[1]);
    for (int y = 0; y < tensor->ne[1]; y++) {
        printf("row %d: ", y);
        for (int x = 0; x < tensor->ne[0]; x++) {
            printf("%.2f ", *(float *) ((char *) tensor->data + y * tensor->nb[1] + x * tensor->nb[0]));
        }
        printf("\n");
    }
    printf("\n");
}

/*
 * This is an example to help understand what ggml_set_rows does.
 *
 * Method signature:
 *   struct ggml_tensor * ggml_set_rows(
 *          struct ggml_context * ctx,
 *          struct ggml_tensor  * a,  // destination tensor (gets modified)
 *          struct ggml_tensor  * b,  // source data (rows to insert)
 *          struct ggml_tensor  * c); // row indices (where to insert)
 */
int main(int argc, char **argv) {
    printf("ggml_set_rows example\n");
    printf("This shows how ggml_set_rows updates specific rows in a destination tensor\n\n");
    
    struct ggml_init_params params = {
        .mem_size   = 16*1024*1024,
        .mem_buffer = NULL,
    };
    struct ggml_context* ctx = ggml_init(params);
    
    const int nx = 2;  // columns
    const int ny = 4;  // rows
    
    struct ggml_tensor* dst = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nx, ny);
    ggml_set_name(dst, "dst");
    
    // Initialize destination with some values
    ggml_set_f32_nd(dst, 0, 0, 0, 0, 10.0f);  // dst[0][0] = 10
    ggml_set_f32_nd(dst, 1, 0, 0, 0, 11.0f);  // dst[0][1] = 11
    ggml_set_f32_nd(dst, 0, 1, 0, 0, 20.0f);  // dst[1][0] = 20
    ggml_set_f32_nd(dst, 1, 1, 0, 0, 21.0f);  // dst[1][1] = 21
    ggml_set_f32_nd(dst, 0, 2, 0, 0, 30.0f);  // dst[2][0] = 30
    ggml_set_f32_nd(dst, 1, 2, 0, 0, 31.0f);  // dst[2][1] = 31
    ggml_set_f32_nd(dst, 0, 3, 0, 0, 40.0f);  // dst[3][0] = 40
    ggml_set_f32_nd(dst, 1, 3, 0, 0, 41.0f);  // dst[3][1] = 41
    
    print_tensor_2d(dst, "Initial state:");
    
    // Create source tensor - this contains the new row data to insert
    const int n_new_rows = 2;  // insert 2 rows
    struct ggml_tensor* source = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nx, n_new_rows);
    ggml_set_name(source, "source");
    
    // New row data to insert
    ggml_set_f32_nd(source, 0, 0, 0, 0, 100.0f);  // new_row_0[0] = 100
    ggml_set_f32_nd(source, 1, 0, 0, 0, 101.0f);  // new_row_0[1] = 101
    ggml_set_f32_nd(source, 0, 1, 0, 0, 200.0f);  // new_row_1[0] = 200
    ggml_set_f32_nd(source, 1, 1, 0, 0, 201.0f);  // new_row_1[1] = 201
    
    print_tensor_2d(source, "source (new row data)");
    
    // Create indices tensor - specifies which rows to update
    struct ggml_tensor* indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_new_rows);
    ggml_set_name(indices, "indices");
    
    // We want to update row 2 and row 0 with our new data
     int64_t* indices_data = (int64_t*)indices->data;
    indices_data[0] = 0;  // source row 0 → destination row 2
    indices_data[1] = 3;  // source row 1 → destination row 0
    
    printf("Row indices to update:\n");
    for (int i = 0; i < indices->ne[0]; i++) {
        printf("source row %d → destination row %d\n", i, (int)indices_data[i]);
    }
    printf("\n");
    
    printf("Performing set_rows operation...\n\n");
    struct ggml_tensor* result = ggml_set_rows(ctx, dst, source, indices);
    ggml_set_name(result, "result");
    
    // Build and execute the computation graph
    struct ggml_cgraph* c_graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(c_graph, result);
    ggml_graph_compute_with_ctx(ctx, c_graph, 1);
    
    print_tensor_2d(result, "Final state:");
    
    ggml_free(ctx);
    return 0;
}
