#include "ggml.h"

#include <stdio.h>

int main(int argc, char **argv) {
  printf("GGML backpropagation example\n");

  struct ggml_init_params params = {
    .mem_size   = 16*1024*1024,
    .mem_buffer = NULL,
  };
  struct ggml_context* ctx = ggml_init(params);

  // 'a' represents a parameter in the graph/neural network
  struct ggml_tensor* a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
  ggml_set_name(a, "a");
  ggml_set_i32_1d(a, 0, 2);
  // Since 'a' is a parameter it's gradient should be stored by calling ggml_set_param.
  ggml_set_param(ctx, a);

  printf("a: %f\n", ggml_get_f32_1d(a, 0));
  printf("a->grad: %s\n", a->grad->name);
  printf("a->grad: %f\n", ggml_get_f32_1d(a->grad, 0));

  // 'b' represents another parameter in the graph/neural network
  struct ggml_tensor* b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
  ggml_set_name(b, "b");
  ggml_set_f32_1d(b, 0, 3);
  // Again since 'b' is a parameter it's gradient should be stored by calling ggml_set_param.
  ggml_set_param(ctx, b);
  printf("b: %f\n", ggml_get_f32_1d(b, 0));
  printf("b->grad: %s\n", b->grad->name);

  struct ggml_tensor* mul = ggml_mul(ctx, a, b);
  printf("mul->grad: %s\n", mul->grad->name);

  struct ggml_cgraph* f_graph = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, true);
  ggml_build_forward_expand(f_graph, mul);

  enum ggml_status st = ggml_graph_compute_with_ctx(ctx, f_graph, 1);
  if (st != GGML_STATUS_SUCCESS) {
    printf("could not compute graph\n");
    return 1;
  }

  printf("a * b = %f\n", ggml_get_f32_1d(mul, 0));

  // Set or the gradients to zero (only really needed if this is not the first
  // backpropagation).
  ggml_graph_reset(f_graph);

  struct ggml_cgraph * b_graph = ggml_graph_dup(ctx, f_graph);
  ggml_build_backward_expand(ctx, f_graph, b_graph, /* keep gradients */ false);

  // Set the gradient of the output tensor (mul) which would be the value of
  // the loss function.
  ggml_set_f32(mul->grad, 2.0f);
  // Compute the gradients
  ggml_graph_compute_with_ctx(ctx, b_graph, 1);

  printf("a->grad: %f\n", ggml_get_f32_1d(a->grad, 0));
  printf("b->grad: %f\n", ggml_get_f32_1d(b->grad, 0));

  // Now, a and b values would be updated using the gradients computed above.
  float learning_rate = 0.01;
  ggml_set_f32_1d(a, 0, ggml_get_f32_1d(a, 0) - learning_rate * ggml_get_f32_1d(a->grad, 0));
  ggml_set_f32_1d(b, 0, ggml_get_f32_1d(b, 0) - learning_rate * ggml_get_f32_1d(b->grad, 0));

  printf("updated parameters a and b:\n");
  printf("a: %f\n", ggml_get_f32_1d(a, 0));
  printf("b: %f\n", ggml_get_f32_1d(b, 0));

  ggml_graph_compute_with_ctx(ctx, f_graph, 1);
  printf("a * b = %f\n", ggml_get_f32_1d(mul, 0));

  ggml_free(ctx);
  return 0;
}
