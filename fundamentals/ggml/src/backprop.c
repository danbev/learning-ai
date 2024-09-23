#include "ggml.h"

#include <stdio.h>

/*
 * This example demonstrates how to perform backpropagation using GGML.
 * This is done manually, with now loops or loss functions caclulated just
 * to understand and explore the inner workings of backpropagation in GGML.
 */
int main(int argc, char **argv) {
  printf("GGML backpropagation example\n");
  printf("\n");


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
  // This will duplicate the tensor in a->grad.
  ggml_set_param(ctx, a);

  printf("Parameter a:\n");
  printf("a: %f\n", ggml_get_f32_1d(a, 0));
  printf("a->grad: %s\n", a->grad->name);
  printf("a->grad: %f\n", ggml_get_f32_1d(a->grad, 0));
  printf("\n");

  // 'b' represents another parameter in the graph/neural network
  struct ggml_tensor* b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
  ggml_set_name(b, "b");
  ggml_set_f32_1d(b, 0, 3);
  // Again since 'b' is a parameter it's gradient should be stored by calling ggml_set_param.
  ggml_set_param(ctx, b);
  printf("Parameter b:\n");
  printf("b: %f\n", ggml_get_f32_1d(b, 0));
  printf("b->grad: %s\n", b->grad->name);
  printf("b->grad: %f\n", ggml_get_f32_1d(b->grad, 0));
  printf("\n");

  printf("Operation/Output tensor mul:\n");
  struct ggml_tensor* mul = ggml_mul(ctx, a, b);
  ggml_set_name(mul, "mul");
  printf("mul->op: %s\n", ggml_op_name(mul->op));
  printf("mul->src0: %s\n", mul->src[0]->name);
  printf("mul->src1: %s\n", mul->src[1]->name);
  printf("mul->grad: %s\n", mul->grad->name);
  printf("\n");

  struct ggml_tensor* five = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
  ggml_set_name(five, "5");
  ggml_set_f32_1d(five, 0, 5);

  struct ggml_cgraph* f_graph = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, true);
  ggml_build_forward_expand(f_graph, five);
  ggml_build_forward_expand(f_graph, mul);
  ggml_graph_print(f_graph);

  printf("[Perform forward pass 1]\n\n");
  enum ggml_status st = ggml_graph_compute_with_ctx(ctx, f_graph, 1);
  if (st != GGML_STATUS_SUCCESS) {
    printf("could not compute graph\n");
    return 1;
  }

  printf("Forward pass 1 result:\n");
  printf("a * b = %f\n", ggml_get_f32_1d(mul, 0));
  printf("\n");

  // Set or the gradients to zero (only really needed if this is not the first
  // backpropagation).
  ggml_graph_reset(f_graph);

  struct ggml_cgraph* b_graph = ggml_graph_dup(ctx, f_graph);
  ggml_build_backward_expand(ctx, f_graph, b_graph, /* keep gradients */ false);
  ggml_graph_print(b_graph);

  // Set the gradient of the output tensor (mul) which would be the value of
  // the loss function.
  ggml_set_f32(mul->grad, 2.0f);
  // Compute the gradients
  printf("[Perform backward pass]\n\n");
  ggml_graph_compute_with_ctx(ctx, b_graph, 1);

  printf("Updated gradients:\n");
  printf("a->grad: %f\n", ggml_get_f32_1d(a->grad, 0));
  printf("b->grad: %f\n", ggml_get_f32_1d(b->grad, 0));
  printf("\n");

  // Now, a and b values would be updated using the gradients computed above.
  float learning_rate = 0.01;
  ggml_set_f32_1d(a, 0, ggml_get_f32_1d(a, 0) - learning_rate * ggml_get_f32_1d(a->grad, 0));
  ggml_set_f32_1d(b, 0, ggml_get_f32_1d(b, 0) - learning_rate * ggml_get_f32_1d(b->grad, 0));

  printf("Updated parameters a and b:\n");
  printf("a: %f\n", ggml_get_f32_1d(a, 0));
  printf("b: %f\n", ggml_get_f32_1d(b, 0));
  printf("\n");

  printf("[Perform forward pass 2]\n\n");
  ggml_graph_compute_with_ctx(ctx, f_graph, 1);

  printf("Forward pass 2 result:\n");
  printf("a * b = %f\n", ggml_get_f32_1d(mul, 0));

  ggml_free(ctx);
  return 0;
}
