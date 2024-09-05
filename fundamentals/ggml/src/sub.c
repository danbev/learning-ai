#include <stdio.h>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

int main(int argc, char **argv) {
  printf("GGML concat examples\n");

  struct ggml_init_params params = {
    .mem_size   = 16*1024*1024,
    .mem_buffer = NULL,
  };
  struct ggml_context* ctx = ggml_init(params);

  struct ggml_tensor* a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5);
  ggml_set_name(a, "a");
  ggml_set_f32_1d(a, 0, 10);
  ggml_set_f32_1d(a, 1, 11);
  ggml_set_f32_1d(a, 2, 12);
  ggml_set_f32_1d(a, 3, 13);
  ggml_set_f32_1d(a, 4, 15);

  printf("a tensor:\n");
  for (int i = 0; i < ggml_nelements(a); i++) {
    printf("a[%d]: %f\n", i, ggml_get_f32_1d(a, i));
  }

  struct ggml_tensor* b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5);
  ggml_set_name(b, "b");
  ggml_set_f32_1d(b, 0, 10);
  ggml_set_f32_1d(b, 1, 10);
  ggml_set_f32_1d(b, 2, 10);
  ggml_set_f32_1d(b, 3, 10);
  ggml_set_f32_1d(b, 4, 10);

  printf("b tensor:\n");
  for (int i = 0; i < ggml_nelements(b); i++) {
    printf("b[%d]: %f\n", i, ggml_get_f32_1d(b, i));
  }

  struct ggml_tensor* result = ggml_sub(ctx, a, b);

  struct ggml_cgraph* c_graph = ggml_new_graph(ctx);
  ggml_build_forward_expand(c_graph, result);
  int n_threads = 4;
  enum ggml_status st = ggml_graph_compute_with_ctx(ctx, c_graph, n_threads);
  if (st != GGML_STATUS_SUCCESS) {
    printf("could not compute graph\n");
    return 1;
  }

  printf("a sub b:\n");
  for (int i = 0; i < ggml_nelements(result); i++) {
    printf("result[%d]: %f\n", i, ggml_get_f32_1d(result, i));
  }

  ggml_free(ctx);
  return 0;
}
