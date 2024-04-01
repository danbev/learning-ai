#include <stdio.h>

#include "ggml/ggml.h"
#include "ggml/ggml-alloc.h"
#include "ggml/ggml-backend.h"

int main(int argc, char **argv) {
  printf("GGML permute tensor examples\n");

  struct ggml_init_params params = {
    .mem_size   = 16*1024*1024,
    .mem_buffer = NULL,
  };
  struct ggml_context* ctx = ggml_init(params);

  const int nx = 2; // x-axis, width of number of columns in the matrix.
  const int ny = 3; // y-axis, the height or the number of rows in the matrix.
  struct ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nx, ny);
  ggml_set_name(a, "a");
  //void* data   = (char *) a->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];

  ggml_set_i32_nd(a, 0, 0, 0, 0, 1);
  ggml_set_i32_nd(a, 1, 0, 0, 0, 2);
  ggml_set_i32_nd(a, 0, 1, 0, 0, 3);
  ggml_set_i32_nd(a, 1, 1, 0, 0, 4);
  ggml_set_i32_nd(a, 0, 2, 0, 0, 5);
  ggml_set_i32_nd(a, 1, 2, 0, 0, 6);

  printf("a ne[0]: %ld\n", a->ne[0]);
  printf("a ne[1]: %ld\n", a->ne[1]);
  printf("a ne[2]: %ld\n", a->ne[2]);
  printf("a ne[3]: %ld\n", a->ne[3]);

  printf("matrix a %ldx%ld:\n", a->ne[0], a->ne[1]);
  for (int y = 0; y < a->ne[1]; y++) {
      for (int x = 0; x < a->ne[0]; x++) {
          printf("%.2f ", *(float *) ((char *) a->data + y * a->nb[1] + x * a->nb[0]));
       }
      printf("\n");
  }
  printf("\n");

  // A permuation is similar to a transpose, but it is a generalization of the
  // transpose operation. The transpose operation is a special case of the
  // permute operation. The first argument is which dimension we want to move
  // or have become the x-axis dimension and the second the which dimension
  // index we want // to move/have as the y-axis dimension.
  //
  // For example, if we want to turn the above 3x2 matrix into a 2x3 matrix we
  // could permute the matrix:
  struct ggml_tensor* p = ggml_permute(ctx, a, 1, 0, 2, 3);
  // This is saying; use dim 1 as the new x-axis and dim 0 as the new y-axis.
  // And keep the other dimensions as they are.
  printf("p ne[0]: %ld\n", p->ne[0]);
  printf("p ne[1]: %ld\n", p->ne[1]);
  printf("p ne[2]: %ld\n", p->ne[2]);
  printf("p ne[3]: %ld\n", p->ne[3]);

  printf("p a %ldx%ld:\n", p->ne[0], p->ne[1]);
  for (int y = 0; y < p->ne[1]; y++) {
      for (int x = 0; x < p->ne[0]; x++) {
          printf("%.2f ", *(float *) ((char *) p->data + y * p->nb[1] + x * p->nb[0]));
       }
      printf("\n");
  }

  ggml_free(ctx);
  return 0;
}
