#include <stdio.h>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

int main(int argc, char **argv) {
  printf("GGML scale tensor examples\n");

  struct ggml_init_params params = {
    .mem_size   = 16*1024*1024,
    .mem_buffer = NULL,
  };
  struct ggml_context* ctx = ggml_init(params);

  struct ggml_tensor* x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10);
  printf("x before graph computation: \n");
  ggml_set_name(x, "x");
  ggml_set_f32_1d(x, 0, 1);
  ggml_set_f32_1d(x, 1, 2);
  ggml_set_f32_1d(x, 2, 3);
  ggml_set_f32_1d(x, 3, 4);
  ggml_set_f32_1d(x, 4, 5);
  ggml_set_f32_1d(x, 5, 6);
  ggml_set_f32_1d(x, 6, 7);
  ggml_set_f32_1d(x, 7, 8);
  ggml_set_f32_1d(x, 8, 9);
  ggml_set_f32_1d(x, 9, 10);

  for (int i = 0; i < ggml_nelements(x); i++) {
    printf("view[%d]: %f\n", i, ggml_get_f32_1d(x, i));
  }

  struct ggml_tensor* s = ggml_scale(ctx, x, 2);

  struct ggml_cgraph* c_graph = ggml_new_graph(ctx);
  ggml_build_forward_expand(c_graph, s);
  ggml_graph_compute_with_ctx(ctx, c_graph, 4);

  printf("x after graph computation: \n");
  for (int i = 0; i < ggml_nelements(s); i++) {
    printf("view[%d]: %f\n", i, ggml_get_f32_1d(s, i));
  }

  ggml_free(ctx);
  return 0;
}
