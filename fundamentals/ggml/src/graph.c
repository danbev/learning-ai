#include <stdio.h>

#include "ggml/ggml.h"
#include "ggml/ggml-alloc.h"
#include "ggml/ggml-backend.h"

int main(int argc, char **argv) {
  printf("GGML compute graph example\n");

  struct ggml_init_params params = {
    .mem_size   = 16*1024*1024,
    .mem_buffer = NULL,
  };
  struct ggml_context* ctx = ggml_init(params);
  printf("ctx mem size: %ld\n", ggml_get_mem_size(ctx));
  printf("ctx mem used: %ld\n", ggml_used_mem(ctx));

  // Create a computation graph (c = computation  in ggml_cgraph)
  struct ggml_cgraph* c_graph = ggml_new_graph(ctx);
  printf("c_graph size: %d\n", c_graph->size);

  // The following will create tensors that will make up the computation graph
  struct ggml_tensor* a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
  ggml_set_name(a, "a");
  struct ggml_tensor* b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
  ggml_set_name(b, "b");
  struct ggml_tensor* c = ggml_add(ctx, a, b);
  printf("c tensor operation: %s, %s\n", ggml_op_name(c->op), ggml_op_symbol(c->op));
  ggml_set_name(c, "c");

  // The following will add the tensors to the computation graph (I think)
  ggml_build_forward_expand(c_graph, c);
  printf("c_graph after build_forward_expand: %d\n", c_graph->size);

  // Now we set the values to be computed
  ggml_set_f32(a, 3.0f);
  ggml_set_f32(b, 2.0f);
  // And finally we compute the values
  ggml_graph_compute_with_ctx(ctx, c_graph, 1);
  printf("c = %f\n", ggml_get_f32_1d(c, 0));
  ggml_graph_dump_dot(c_graph, NULL, "add.dot");

  ggml_free(ctx);
  return 0;
}
