#include "ggml.h"
#include "ggml-cpu.h"

#include <stdio.h>

int main(int argc, char **argv) {
  printf("GGML ggml_out_prod example\n");
  printf("\n");

  struct ggml_init_params params = {
    .mem_size   = 16*1024*1024,
    .mem_buffer = NULL,
  };
  struct ggml_context* ctx = ggml_init(params);

  struct ggml_tensor* a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2);
  ggml_set_name(a, "a");
  ggml_set_i32_1d(a, 0, 1);
  ggml_set_i32_1d(a, 1, 2);
  ggml_set_param(ctx, a);

  struct ggml_tensor* b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2);
  ggml_set_name(b, "b");
  ggml_set_f32_1d(b, 0, 3);
  ggml_set_f32_1d(b, 1, 4);
  ggml_set_param(ctx, b);

  struct ggml_tensor* out = ggml_out_prod(ctx, a, b);
  ggml_set_name(out, "out_prod");
  printf("out->src0: %s\n", out->src[0]->name);
  printf("out->src1: %s\n", out->src[1]->name);
  printf("out->ne[0]: %ld\n", out->ne[0]);
  printf("out->ne[1]: %ld\n", out->ne[1]);
  printf("\n");

  struct ggml_cgraph* f_graph = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, true);
  ggml_build_forward_expand(f_graph, out);
  ggml_graph_print(f_graph);

  printf("[Perform forward pass 1]\n\n");
  enum ggml_status st = ggml_graph_compute_with_ctx(ctx, f_graph, 1);
  if (st != GGML_STATUS_SUCCESS) {
    printf("could not compute graph\n");
    return 1;
  }

  printf("Forward pass 1 result:\n");
  printf("out n_dims: %d\n", ggml_n_dims(out));
  printf("[%.01f  %.01f]\n", ggml_get_f32_nd(out, 0, 0, 0, 0), ggml_get_f32_nd(out, 0, 1, 0, 0));
  printf("[%.01f  %.01f]\n", ggml_get_f32_nd(out, 1, 0, 0, 0), ggml_get_f32_nd(out, 1, 1, 0, 0));
  printf("\n");

  ggml_free(ctx);
  return 0;
}
