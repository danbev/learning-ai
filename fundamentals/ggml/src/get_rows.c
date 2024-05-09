#include <stdio.h>

#include "ggml/ggml.h"
#include "ggml/ggml-alloc.h"
#include "ggml/ggml-backend.h"

/*
 * This is an example to help understand what ggml_get_rows does.
 */
int main(int argc, char **argv) {
  printf("GGML get_rows example\n");

  struct ggml_init_params params = {
    .mem_size   = 16*1024*1024,
    .mem_buffer = NULL,
  };
  struct ggml_context* ctx = ggml_init(params);

  // This tensor will act as the tensor that we want to extract from.
  const int nx = 2; // x-axis, width or number of columns in the matrix.
  const int ny = 3; // y-axis, the height or the number of rows in the matrix.
  struct ggml_tensor* a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, nx, ny, 1);
  ggml_set_name(a, "a");
  //void* data   = (char *) a->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];

  ggml_set_i32_nd(a, 0, 0, 0, 0, 1);
  ggml_set_i32_nd(a, 1, 0, 0, 0, 2);
  ggml_set_i32_nd(a, 0, 1, 0, 0, 3);
  ggml_set_i32_nd(a, 1, 1, 0, 0, 4);
  ggml_set_i32_nd(a, 0, 2, 0, 0, 5);
  ggml_set_i32_nd(a, 1, 2, 0, 0, 6);

  printf("a tensor n_dims: %d\n", ggml_n_dims(a));
  for (int y = 0; y < ny; y++) {
      for (int x = 0; x < nx; x++) {
          printf("%.2f ", *(float *) ((char *) a->data + y * a->nb[1] + x * a->nb[0]));
       }
      printf("\n");
  }
  printf("\n");

  // This tensor will act as the tensor which specifies which indices to extract
  // rows from the tensor a. So each entry in indices will specify a row of
  // a that will be extracted.
  struct ggml_tensor* indices = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, 1);
  ggml_set_name(indices, "indices");
  printf("indices tensor n_dims: %d\n", ggml_n_dims(indices));

  // Specify that we want to extract the second row.
  ggml_set_i32_1d(indices, 0, 2);
  // Specify that we want to extract the first row.
  ggml_set_i32_1d(indices, 1, 0);
  printf("row indices to extract/get:\n");
  for (int i = 0; i < indices->ne[0]; i++) {
      printf("%d ", ggml_get_i32_1d(indices, i));
  }
  printf("\n\n");

  struct ggml_tensor* rows = ggml_get_rows(ctx, a, indices);
  ggml_set_name(rows, "rows");

  struct ggml_cgraph* c_graph = ggml_new_graph(ctx);
  ggml_build_forward_expand(c_graph, rows);
  ggml_graph_compute_with_ctx(ctx, c_graph, 1);

  printf("rows tensor dimensions: %d\n", ggml_n_dims(rows));
  // Recall that we have a 3x2 matrix (3 rows and 2 columns).
  //  ne[1]   1.00  2.00
  //   ↓      3.00  4.00
  //          5.00  6.00
  //
  //            ne[0] →
  // So below we want to loop over the rows, so that is ne[1], and then over
  // each column which is ne[0].
  for (int y = 0; y < rows->ne[1]; y++) {
      printf("row %d: ", y);
      for (int x = 0; x < rows->ne[0]; x++) {
          printf("%.2f ", *(float *) ((char *) rows->data + y * rows->nb[1] + x * rows->nb[0]));
       }
      printf("\n");
  }
  printf("\n");

  ggml_graph_dump_dot(c_graph, NULL, "get-rows.dot");

  //ggml_print_objects(ctx);

  ggml_free(ctx);
  return 0;
}
