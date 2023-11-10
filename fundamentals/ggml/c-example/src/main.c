#include <stdio.h>

#include "ggml/ggml.h"
#include "ggml/ggml-alloc.h"

int main(int argc, char **argv) {
  printf("GGML Example\n");

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
  printf("x tensor backend: %d (0 = GGML_BACKEND_CPU) \n", x->backend);
  printf("x tensor dimensions: %d\n", x->n_dims);
  printf("x tensor data: %p\n", x->data);
  // This tensor was not created by an operation, for example if the tensor was
  // created by a + b = c, c being the tensor then the op would be GGML_OP_ADD.
  printf("x tensor operation: %d\n", x->op);
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

  // Create a computation graph (c = computation)
  struct ggml_cgraph* graph = ggml_new_graph(ctx);
  //ggml_set_param(ctx, x);

  ggml_free(ctx);
  return 0;
}
