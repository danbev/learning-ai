#include <stdio.h>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

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
  struct ggml_tensor* x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10);
  ggml_set_name(x, "x");
  printf("x tensor type: %s\n", ggml_type_name(x->type));
  printf("x tensor backend: %d \n", x->backend);
  printf("x tensor dimensions: %d\n", ggml_n_dims(x));
  printf("x tensor data: %p\n", x->data);
  printf("x tensor ne[0]: %ld\n", x->ne[0]);
  printf("x tensor nb[0]: %ld\n", x->nb[0]);
  printf("x tensor nb[1]: %ld\n", x->nb[1]);
  // This tensor was not created by an operation, for example if the tensor was
  // created by a + b = c, c being the tensor then the op would be GGML_OP_ADD.
  printf("x tensor operation: %s, %s\n", ggml_op_name(x->op), ggml_op_symbol(x->op));
  // ggml_tensor's are used as the base unit values in the library, similar to
  // the Value struct in the LLM zero-to-hero tutorial. These values support
  // automatic differentiation, so they have a grad field. 
  printf("x tensor grad: %p\n", x->grad);
  // src are the values that were used to create the tensor, for example if the
  // tensor was created by a + b = c, then the src would be a and b.
  printf("x tensor src: %p\n", x->src);
  printf("x tensor name: %s\n", x->name);
  // The following I'm guessing is a flag to indicate if this tensor should be
  // taking part in the automatic differentiation process or not.
  printf("x tensor flags: %d\n", x->flags);

  // Example of updating a tensor:
  // Note that ggml_set_i32 will set all the values in the tensor to 18, which
  // in this case will be 10.
  struct ggml_tensor* updated = ggml_set_i32(x, 18);
  printf("updated tensor data: %lf\n", *ggml_get_data_f32(updated));
  printf("updated[0]=: %lf\n", ggml_get_f32_1d(updated, 0));
  printf("updated[9]=: %lf\n", ggml_get_f32_1d(updated, 9));

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
          printf("value: %p\n", value);
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
  printf("Nr of elements in 1 dim (ne[0]): %ld\n", matrix->ne[0]);
  printf("Nr of elements in 2 dim (ne[1]): %ld\n", matrix->ne[1]);

  printf("stride for 1 dim (nb[0]): %ld (ggml_type_size: %ld)\n", matrix->nb[0], ggml_type_size(matrix->type));
  printf("stride for 2 dim (nb[1]): %ld (%ld * %ld / %d) + paddings  \n", matrix->nb[1],
                                              matrix->nb[0],
                                              matrix->ne[0],
                                              ggml_blck_size(matrix->type));
  printf("matrix name: %s\n", ggml_get_name(matrix));
  ggml_print_objects(ctx);

  //struct ggml_tensor* y = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10);
  //ggml_set_name(y, "y");
  struct ggml_tensor* dup = ggml_dup_tensor(ctx, x);
  printf("dup tensor type: %s\n", ggml_type_name(dup->type));
  printf("dup tensor dimensions: %d\n", ggml_n_dims(dup));
  printf("x tensor ne[0]: %ld\n", dup->ne[0]);
  printf("x tensor n3[1]: %ld\n", dup->ne[1]);
  printf("x tensor n3[2]: %ld\n", dup->ne[2]);
  printf("x tensor n3[3]: %ld\n", dup->ne[3]);
  // Name is not duplicated
  printf("dup tensor name: %s\n", dup->name);

  ggml_free(ctx);
  return 0;
}
