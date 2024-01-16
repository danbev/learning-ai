#include <stdio.h>

#include "ggml/ggml.h"
#include "ggml/ggml-alloc.h"
#include "ggml/ggml-backend.h"

int main(int argc, char **argv) {
  printf("GGML tensor example\n");

  struct ggml_init_params params = {
    .mem_size   = 16*1024*1024,
    .mem_buffer = NULL,
  };
  struct ggml_context* ctx = ggml_init(params);

  const int nx = 2; // x-axis, width of number of columns in the matrix.
  const int ny = 3; // y-axis, the height or the number of rows in the matrix.
  struct ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nx, ny);
  ggml_set_name(a, "a");
  *(float *)( (char *) a->data) = 1;
  *(float *)( (char *) a->data + 4) = 2;
  *(float *)( (char *) a->data + 8) = 3;
  *(float *)( (char *) a->data + 12) = 4;
  *(float *)( (char *) a->data + 16) = 5;
  *(float *)( (char *) a->data + 20) = 6;

  printf("matrix nb[0] type: %ld\n", a->nb[0]);
  printf("matrix nb[1] type: %ld\n", a->nb[1]);

  printf("matrix a %ldx%ld:\n", a->ne[0], a->ne[1]);
  for (int y = 0; y < ny; y++) {
      for (int x = 0; x < nx; x++) {
          printf("%.2f ", *(float *) ((char *) a->data + y * a->nb[1] + x * a->nb[0]));
       }
      printf("\n");
  }
  printf("\n");

  struct ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 1);
  ggml_set_name(b, "b");
  *(float *)( (char *) b->data) = 7;
  *(float *)( (char *) b->data + 4) = 8;

  // Note that ggml_mul_mat() transpose matrix b.
  struct ggml_tensor* result = ggml_mul_mat(ctx, a, b);
  ggml_set_name(result, "result");
  /*
       
    +---+---+    +---+---+T    +---+---+   +---+    +---+
    | 1 | 2 |    | 7 | 8 |  =  | 1 | 2 |   | 7 |    | 23|
    +---+---+ X  +---+---+     +---+---+ X +---+ =  +---+
    | 3 | 4 |                  | 3 | 4 |   | 8 |    | 53|
    +---+---+                  +---+---+   +---+    +---+
    | 5 | 6 |                  | 5 | 6 |            | 83|
    +---+---+                  +---+---+            +---+
   */


  struct ggml_cgraph* c_graph = ggml_new_graph(ctx);
  ggml_build_forward_expand(c_graph, result);
  ggml_graph_compute_with_ctx(ctx, c_graph, 4);

  printf("result tensor type: %s\n", ggml_type_name(result->type));
  printf("result dim: %d\n", result->n_dims);
  printf("result dim[0]: %ld\n", result->ne[0]);
  printf("result dim[1]: %ld\n", result->ne[1]);

  printf("\n");
  printf("result matrix %ldx%ld:\n", result->ne[0], result->ne[1]);
  for (int y = 0; y < result->ne[1]; y++) {
      for (int x = 0; x < result->ne[0]; x++) {
          printf("%.2f ", *(float *) ((char *) result->data + y * result->nb[1] + x * result->nb[0]));
       }
      printf("\n");
  }
  printf("\n");

  ggml_graph_dump_dot(c_graph, NULL, "mul.dot");

  ggml_free(ctx);
  return 0;
}
