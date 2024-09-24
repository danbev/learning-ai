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

  // 'b' represents another parameter in the graph/neural network
  struct ggml_tensor* b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 3, 1, 2, 1);
  ggml_set_name(b, "b");

  printf("Operation/Output tensor mul:\n");
  struct ggml_tensor* mul = ggml_mul(ctx, a, b);
  ggml_set_name(mul, "mul");

  struct ggml_cgraph* f_graph = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, true);
  ggml_build_forward_expand(f_graph, mul);
  ggml_graph_print(f_graph);

  printf("[Perform forward pass 1]\n\n");
  enum ggml_status st = ggml_graph_compute_with_ctx(ctx, f_graph, 1);
  if (st != GGML_STATUS_SUCCESS) {
    printf("could not compute graph\n");
    return 1;
  }

  ggml_free(ctx);
  return 0;
}
