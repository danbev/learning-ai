#include <stdio.h>

#include "ggml.h"
#include "ggml-cpu.h"

int main(int argc, char **argv) {
  printf("GGML accumulate example\n");

  struct ggml_init_params params = {
    .mem_size   = 16*1024*1024,
    .mem_buffer = NULL,
  };
  struct ggml_context* ctx = ggml_init(params);

  // This is the destination tensor
  struct ggml_tensor* a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5);
  ggml_set_name(a, "a");
  ggml_set_f32_1d(a, 0, 1);
  ggml_set_f32_1d(a, 1, 2);
  ggml_set_f32_1d(a, 2, 3);
  ggml_set_f32_1d(a, 3, 4);
  ggml_set_f32_1d(a, 4, 5);

  printf("a tensor:\n");
  for (int i = 0; i < ggml_nelements(a); i++) {
    printf("a[%d]: %f\n", i, ggml_get_f32_1d(a, i));
  }

  // The source tensor which will have its values added to the destination
  // tensor dimensions specified later in the ggml_acc call.
  struct ggml_tensor* b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5);
  ggml_set_name(b, "b");
  ggml_set_f32_1d(b, 0, 6);
  ggml_set_f32_1d(b, 1, 7);
  ggml_set_f32_1d(b, 2, 8);
  ggml_set_f32_1d(b, 3, 9);
  ggml_set_f32_1d(b, 4, 10);

  printf("b tensor:\n");
  for (int i = 0; i < ggml_nelements(b); i++) {
    printf("b[%d]: %f\n", i, ggml_get_f32_1d(b, i));
  }

  struct ggml_tensor* result = ggml_acc(ctx, a, b, 1, 1, 1, 0);

  struct ggml_cgraph* c_graph = ggml_new_graph(ctx);
  ggml_build_forward_expand(c_graph, result);
  int n_threads = 4;
  enum ggml_status st = ggml_graph_compute_with_ctx(ctx, c_graph, n_threads);
  if (st != GGML_STATUS_SUCCESS) {
    printf("could not compute graph\n");
    return 1;
  }

  printf("result:\n");
  for (int i = 0; i < ggml_nelements(result); i++) {
    printf("result[%d]: %f\n", i, ggml_get_f32_1d(result, i));
  }

  ggml_free(ctx);
  return 0;
}
