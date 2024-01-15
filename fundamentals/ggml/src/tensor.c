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
  printf("ctx mem size: %ld\n", ggml_get_mem_size(ctx));
  printf("ctx mem used: %ld\n", ggml_used_mem(ctx));

  // This creates a one dimensional tensor, so it will be like a list of numbers
  struct ggml_tensor* x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
  printf("x tensor type: %s\n", ggml_type_name(x->type));
  printf("x tensor backend: %d \n", x->backend);
  printf("x tensor dimensions: %d\n", x->n_dims);
  printf("x tensor data: %p\n", x->data);
  printf("x tensor ne[0]: %ld\n", x->ne[0]);
  printf("x tensor nb[0]: %ld\n", x->nb[0]);
  printf("x tensor nb[1]: %ld\n", x->nb[1]);
  // This tensor was not created by an operation, for example if the tensor was
  // created by a + b = c, c being the tensor then the op would be GGML_OP_ADD.
  printf("x tensor operation: %s, %s\n", ggml_op_name(x->op), ggml_op_symbol(x->op));
  // ggml_tensor's are used as the base unit values in the library, similar to
  // the Value struct in the LLM zero-to-hero tutorial. These values support
  // autmoatic differentiation, so they have a grad field. 
  printf("x tensor grad: %p\n", x->grad);
  // src are the values that were used to create the tensor, for example if the
  // tensor was created by a + b = c, then the src would be a and b.
  printf("x tensor src: %p\n", x->src);
  printf("x tensor name: %s\n", x->name);
  // The following I'm guessing is a flag to indicate if this tensor should be
  // taking part in the automatic differentiation process or not.
  printf("x tensor is_param: %d\n", x->is_param);

  // Example of updating a tensor:
  struct ggml_tensor* updated = ggml_set_i32(x, 18);
  printf("updated tensor data: %lf\n", *ggml_get_data_f32(updated));

  ggml_set_name(updated, "updated");
  printf("updated tensor name: %s\n", ggml_get_name(updated));

  printf("\n\n");

  const int nx = 3; // x-axis, width of number of columns in the matrix.
  const int ny = 2; // y-axis, the height or the number of rows in the matrix.
  //  +---+---+---+
  //  |   |   |   |
  //  +---+---+---+
  //  |   |   |   |
  //  +---+---+---+
  //
  struct ggml_tensor* matrix = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nx, ny);
  int v = 0;

  for (int y = 0; y < ny; y++) {
      for (int x = 0; x < nx; x++) {
          float* value = (float *)( (char *) matrix->data + (y * matrix->nb[1]) + (x * matrix->nb[0]));
          // Since we want to perform byte-level operations and a char is 1 byte.
          // If we don't do this the additions would be done of the size of the
          // type, so 4 bytes for a float. And after we are done we need to
          // cast back to a float pointer.
          *value = v;
          v++;
       }
  }

  printf("matrix 3x2:\n");
  for (int y = 0; y < ny; y++) {
      for (int x = 0; x < nx; x++) {
          printf("%.2f ", *(float *) ((char *) matrix->data + y * matrix->nb[1] + x * matrix->nb[0]));
       }
      printf("\n");
  }
  printf("\n");

  //  +---+---+---+
  //  | 0 | 1 | 2 | nb[0] = 4
  //  +---+---+---+
  //  | 3 | 4 | 5 | nb[1] = 12
  //  +---+---+---+
  //
  // Memory layout:
  // 0000 0001 0010   0011 0100 0101
  //   0    1   2      3    4    5
  //    row 1              row 2
  //
  // 24 bytes in total
  printf("elements in 1 dim: %ld\n", matrix->ne[0]);
  printf("elements in 2 dim: %ld\n", matrix->ne[1]);

  printf("stride for 1 dim: %ld (ggml_type_size: %ld)\n", matrix->nb[0], ggml_type_size(matrix->type));
  printf("stride for 2 dim: %ld (%ld * %ld / %d) + paddings  \n", matrix->nb[1],
                                              matrix->nb[0],
                                              matrix->ne[0],
                                              ggml_blck_size(matrix->type));
  printf("matrix name: %s\n", ggml_get_name(matrix));
  ggml_print_objects(ctx);

  ggml_free(ctx);
  return 0;
}
