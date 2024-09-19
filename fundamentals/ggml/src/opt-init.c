#include <stdio.h>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

int main(int argc, char **argv) {
  printf("GGML optimizer example\n");

  struct ggml_init_params params = {
    .mem_size   = 16*1024*1024,
    .mem_buffer = NULL,
  };
  struct ggml_context* ctx = ggml_init(params);

  struct ggml_opt_params opts = ggml_opt_default_params(GGML_OPT_TYPE_ADAM);
  struct ggml_tensor* a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
  ggml_set_param(ctx, a);

  struct ggml_cgraph * cgraph = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, true);
  ggml_build_forward_expand(cgraph, a);

  ggml_opt(ctx, opts, a);

  ggml_graph_compute_with_ctx(ctx, cgraph, 1);

  printf("a: n_elements: %ld\n", ggml_nelements(a));

  ggml_free(ctx);
  return 0;
}
