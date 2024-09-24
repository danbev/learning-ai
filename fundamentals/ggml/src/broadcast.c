#include "ggml.h"

#include <stdio.h>

int main(int argc, char **argv) {
  printf("GGML broadcast example\n");
  printf("\n");


  struct ggml_init_params params = {
    .mem_size   = 16*1024*1024,
    .mem_buffer = NULL,
  };
  struct ggml_context* ctx = ggml_init(params);

  struct ggml_tensor* a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 3, 2, 2, 2);
  ggml_set_name(a, "a");
  printf("a n_element: %ld\n", ggml_nelements(a));
  float* a_data = (float*)a->data;
  for (int i = 0; i < 24; i++) {
        a_data[i] = i + 1;
  }
  for (int i = 0; i < ggml_nelements(a); i++) {
    printf("a[%d]: %f\n", i, ggml_get_f32_1d(a, i));
  }
  printf("\n");

  // In this case we should see broadcasting happen for the y and z dimensions.
  struct ggml_tensor* b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 3, 1, 2, 1);
  ggml_set_name(b, "b");
  float* b_data = (float*) b->data;
  for (int i = 0; i < 6; i++) {
        b_data[i] = i + 1;
  }
  printf("b n_element: %ld\n", ggml_nelements(b));
  for (int i = 0; i < ggml_nelements(b); i++) {
    printf("b[%d]: %f\n", i, ggml_get_f32_1d(b, i));
  }
  printf("\n");

  struct ggml_tensor* mul = ggml_mul(ctx, a, b);
  ggml_set_name(mul, "mul");

  struct ggml_cgraph* f_graph = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, true);
  ggml_build_forward_expand(f_graph, mul);

  enum ggml_status st = ggml_graph_compute_with_ctx(ctx, f_graph, 1);
  if (st != GGML_STATUS_SUCCESS) {
    printf("could not compute graph\n");
    return 1;
  }

  printf("Multiplication result:\n");
  for (int i = 0; i < ggml_nelements(mul); i++) {
    printf("mul[%d]: %f\n", i, ggml_get_f32_1d(mul, i));
  }
  printf("\n");


  ggml_free(ctx);
  return 0;
}
