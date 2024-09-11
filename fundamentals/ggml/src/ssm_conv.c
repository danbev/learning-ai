#include <stdio.h>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

int main(int argc, char **argv) {
  printf("GGML ssm_conv example\n");

  struct ggml_init_params params = {
    .mem_size   = 16*1024*1024,
    .mem_buffer = NULL,
  };
  struct ggml_context* ctx = ggml_init(params);

  // This represents the token embeddings after they have been projected to
  // a higher dimension in a Mamba block. 
  int d_conv = 3;           // Size of the convolutional filter.
  int n_tokens = 4;         // Number of tokens in the sequence.
  int d_inner = 16;         // Dimension of the inner layer.
  int padded_length = n_tokens + d_conv - 1;
  printf("d_conv: %d:\n", d_conv);
  printf("n_tokens: %d:\n", n_tokens);
  printf("d_inner: %d:\n", d_inner);
  printf("padded_length: %d:\n", padded_length);
                                            
  struct ggml_tensor* sx = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
          padded_length,
          d_inner,
          1);
  printf("sx n_dims: %d:\n", ggml_n_dims(sx));
  printf("sx nelements: %ld:\n", ggml_nelements(sx));
  printf("sx->ne[0]: %ld:\n", sx->ne[0]);
  printf("sx->ne[1]: %ld:\n", sx->ne[1]);
  // Populate the tensor with some values.
  for (int i = 0; i < ggml_nelements(sx); i++) {
      ggml_set_f32_nd(sx, i, 0, 0, 0, (float)i);
  }
  /*
  for (int i = 0; i < ggml_nelements(sx); i++) {
    printf("a[%d]: %0.2f\n", i, ggml_get_f32_1d(sx, i));
  }
  */

  // This is our convolutional filter. 
  struct ggml_tensor* c = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_conv, d_inner);
  printf("c->ne[0]: %ld:\n", c->ne[0]); // is d_conv value.
  printf("c->ne[1]: %ld:\n", c->ne[1]); // is d_inner value.
  struct ggml_tensor* result = ggml_ssm_conv(ctx, sx, c);
  for (int i = 0; i < ggml_nelements(c); i++) {
      ggml_set_f32_nd(c, i, 0, 0, 0, (float)i);
  }

  struct ggml_cgraph* c_graph = ggml_new_graph(ctx);
  ggml_build_forward_expand(c_graph, result);

  int n_threads = 4;
  enum ggml_status st = ggml_graph_compute_with_ctx(ctx, c_graph, n_threads);
  if (st != GGML_STATUS_SUCCESS) {
    printf("could not compute graph\n");
    return 1;
  }

  printf("result tensor nelements: %ld:\n", ggml_nelements(result));
  printf("result tensor n_dims: %d:\n", ggml_n_dims(result));
  /*
  for (int i = 0; i < ggml_nelements(result); i++) {
    printf("a[%d]: %0.2f\n", i, ggml_get_f32_1d(result, i));
  }
  */

  ggml_free(ctx);
  return 0;
}

