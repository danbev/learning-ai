#include <stdio.h>
#include <string.h>

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

/*
 * A 1d convolution is a convolution operation that is applied to a
 * one-dimensional input, such as a sequence of numbers or a time series.
 *
 * But the input to this operation is not a 1d tensor.
 */
int main(int argc, char **argv) {
  printf("GGML conv_1d example\n\n");

  struct ggml_init_params params = {
    .mem_size   = 16*1024*1024,
    .mem_buffer = NULL,
  };
  struct ggml_context* ctx = ggml_init(params);

  struct ggml_tensor* a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3);
  ggml_set_name(a, "a");
  ggml_set_i32_nd(a, 0, 0, 0, 0, 1);
  ggml_set_i32_nd(a, 1, 0, 0, 0, 2);
  ggml_set_i32_nd(a, 0, 1, 0, 0, 3);

  printf("a (convolution kernel)\n");
  for (int y = 0; y < a->ne[1]; y++) {
      for (int x = 0; x < a->ne[0]; x++) {
          printf("%.2f ", *(float *) ((char *) a->data + y * a->nb[1] + x * a->nb[0]));
       }
      printf("\n");
  }
  printf("\n");

  struct ggml_tensor* b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 6);
  ggml_set_name(b, "b");
  ggml_set_i32_nd(b, 0, 0, 0, 0, 1);
  ggml_set_i32_nd(b, 1, 0, 0, 0, 2);
  ggml_set_i32_nd(b, 2, 0, 0, 0, 3);
  ggml_set_i32_nd(b, 3, 0, 0, 0, 4);
  ggml_set_i32_nd(b, 4, 0, 0, 0, 5);
  ggml_set_i32_nd(b, 5, 0, 0, 0, 6);

  printf("b (data):\n");
  for (int y = 0; y < b->ne[1]; y++) {
      for (int x = 0; x < b->ne[0]; x++) {
          printf("%.2f ", *(float *) ((char *) b->data + y * b->nb[1] + x * b->nb[0]));
       }
      printf("\n");
  }
  printf("\n");

  int stride = 1;
  int pad = 0;
  // Dilation ("atrous convoluation" form the French term "with holes") rate.
  // 0 means no kernel elementes will be applied.
  // 1 means no dilation.
  // 2 means one space between kernel elements.
  int dil = 1;
  struct ggml_tensor* result = ggml_conv_1d(ctx, a, b, stride, pad, dil);

  struct ggml_cgraph* c_graph = ggml_new_graph(ctx);
  ggml_build_forward_expand(c_graph, result);
  ggml_graph_compute_with_ctx(ctx, c_graph, 1);

  printf("result:\n");
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
