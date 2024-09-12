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
  int padding = d_conv -1;
  int padded_length = n_tokens + padding;
  printf("d_conv: %d:\n", d_conv);
  printf("n_tokens: %d:\n", n_tokens);
  printf("d_inner: %d:\n", d_inner);
  printf("padding: %d:\n", padding);
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
  ggml_set_f32_nd(sx, 2, 0, 0, 0, 1);
  ggml_set_f32_nd(sx, 3, 0, 0, 0, 2);
  ggml_set_f32_nd(sx, 4, 0, 0, 0, 3);
  ggml_set_f32_nd(sx, 5, 0, 0, 0, 4);
  printf("\nsx tensor (input) to the convolutional layer:\n");
  for (int y = 0; y < sx->ne[1]; y++) {
      for (int x = 0; x < sx->ne[0]; x++) {
          printf("%.2f ", *(float *) ((char *) sx->data + y * sx->nb[1] + x * sx->nb[0]));
       }
      printf("\n");
  }
  printf("\n");

  // This is our convolutional filter. 
  struct ggml_tensor* c = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_conv, d_inner);
  printf("c->ne[0]: %ld:\n", c->ne[0]); // is d_conv value.
  printf("c->ne[1]: %ld:\n", c->ne[1]); // is d_inner value.
                                        
  struct ggml_tensor* result = ggml_ssm_conv(ctx, sx, c);
  ggml_set_f32_nd(c, 0, 0, 0, 0, 1);
  ggml_set_f32_nd(c, 1, 0, 0, 0, 2);
  ggml_set_f32_nd(c, 2, 0, 0, 0, 3);

  printf("\nc tensor (kernel):\n");
  for (int y = 0; y < c->ne[1]; y++) {
      for (int x = 0; x < c->ne[0]; x++) {
          printf("%.2f ", *(float *) ((char *) c->data + y * c->nb[1] + x * c->nb[0]));
       }
      printf("\n");
  }
  printf("\n");

  struct ggml_cgraph* c_graph = ggml_new_graph(ctx);
  ggml_build_forward_expand(c_graph, result);

  int n_threads = 1;
  enum ggml_status st = ggml_graph_compute_with_ctx(ctx, c_graph, n_threads);
  if (st != GGML_STATUS_SUCCESS) {
    printf("could not compute graph\n");
    return 1;
  }

  printf("result->ne[0]: %ld:\n", result->ne[0]);
  printf("result->ne[1]: %ld:\n", result->ne[1]);
  printf("result tensor nelements: %ld:\n", ggml_nelements(result));
  printf("\nresult of convolution:\n");
  for (int y = 0; y < result->ne[1]; y++) {
      for (int x = 0; x < result->ne[0]; x++) {
          printf("%.2f ", *(float *) ((char *) result->data + y * result->nb[1] + x * result->nb[0]));
       }
      printf("\n");
  }
  printf("\n");

  ggml_free(ctx);
  return 0;
}

