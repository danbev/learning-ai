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

  // tensors are stored in row-major order which means that they are layed out
  // row after row in memory: [ [ 1, 2 ],
  //                            [ 3, 4 ],
  //                            [ 5, 6 ] ]
  // [1, 2, 3, 4, 5, 6]
  struct ggml_tensor* matrix = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 2);
  printf("matrix ne[0]: %ld\n", matrix->ne[0]);
  printf("matrix ne[1]: %ld\n", matrix->ne[1]);
  // ne[0] is the number of bytes to move to get to the next element in a row.
  printf("matrix nb[0]: %ld\n", matrix->nb[0]);
  // ne[1] is the number of bytes to move to get to the next row.
  printf("matrix nb[1]: %ld\n", matrix->nb[1]);
  // So we have 4 bytes be value and 12 bytes per row.
  // [1, 2, 3, 4, 5, 6]
  // ne[0] = 3, ne[1] = 2
  // nb[0] = 4, nb[1] = 12
  // result->nb[1] = result->nb[0]*(result->ne[0]/ggml_blck_size(type));
  // result->nb[1] = 4 * (3/1) = 12
  // 
  //    1    2        3    4        5   6
  // 0            12            24            36
  //      row0        row1            row2

  printf("matrix name: %s\n", ggml_get_name(matrix));
  ggml_free(ctx);
  return 0;
}
