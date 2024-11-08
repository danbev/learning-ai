#include <stdio.h>
#include <string.h>

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

int main(int argc, char **argv) {
  printf("GGML conv_2d example\n\n");

  struct ggml_init_params params = {
    .mem_size   = 16*1024*1024,
    .mem_buffer = NULL,
  };
  struct ggml_context* ctx = ggml_init(params);

  struct ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 2);
  ggml_set_name(a, "a");
  ggml_set_i32_nd(a, 0, 0, 0, 0, 1);
  ggml_set_i32_nd(a, 1, 0, 0, 0, 2);
  ggml_set_i32_nd(a, 0, 1, 0, 0, 3);
  ggml_set_i32_nd(a, 1, 1, 0, 0, 4);

  printf("a (convolution kernel)\n");
  for (int y = 0; y < a->ne[1]; y++) {
      for (int x = 0; x < a->ne[0]; x++) {
          printf("%.2f ", *(float *) ((char *) a->data + y * a->nb[1] + x * a->nb[0]));
       }
      printf("\n");
  }
  printf("\n");

  struct ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 2);
  ggml_set_name(b, "b");
  ggml_set_i32_nd(b, 0, 0, 0, 0, 1);
  ggml_set_i32_nd(b, 1, 0, 0, 0, 2);
  ggml_set_i32_nd(b, 2, 0, 0, 0, 3);
  ggml_set_i32_nd(b, 3, 0, 0, 0, 4);
  ggml_set_i32_nd(b, 4, 0, 0, 0, 5);
  ggml_set_i32_nd(b, 5, 0, 0, 0, 6);
  ggml_set_i32_nd(b, 6, 0, 0, 0, 7);
  ggml_set_i32_nd(b, 7, 0, 0, 0, 8);
  ggml_set_i32_nd(b, 0, 1, 0, 0, 9);
  ggml_set_i32_nd(b, 1, 1, 0, 0, 10);
  ggml_set_i32_nd(b, 2, 1, 0, 0, 11);
  ggml_set_i32_nd(b, 3, 1, 0, 0, 12);
  ggml_set_i32_nd(b, 4, 1, 0, 0, 13);
  ggml_set_i32_nd(b, 5, 1, 0, 0, 14);
  ggml_set_i32_nd(b, 6, 1, 0, 0, 15);
  ggml_set_i32_nd(b, 7, 1, 0, 0, 16);

  printf("b (data):\n");
  for (int y = 0; y < b->ne[1]; y++) {
      for (int x = 0; x < b->ne[0]; x++) {
          printf("%.2f ", *(float *) ((char *) b->data + y * b->nb[1] + x * b->nb[0]));
       }
      printf("\n");
  }
  printf("\n");

  int stride_x = 1;
  int stride_y = 1;
  int pad_x = 0;
  int pad_y = 0;
  // Dilation ("atrous convoluation" form the French term "with holes") rate.
  // 0 means no kernel elementes will be applied.
  // 1 means no dilation.
  // 2 means one space between kernel elements.
  int dil_x = 1;
  int dil_y = 1;
  struct ggml_tensor* result = ggml_conv_2d(ctx, a, b, stride_x, stride_y, pad_x, pad_y, dil_x, dil_y);

  struct ggml_cgraph* c_graph = ggml_new_graph(ctx);
  ggml_build_forward_expand(c_graph, result);
  ggml_graph_compute_with_ctx(ctx, c_graph, 1);

  // 0; [1*1 + 2*2 + 3*9 + 4*10] = 1 + 4 + 27 + 40 = 72
  // 1; [1*2 + 2*3 + 3*10 + 4*11] = 2 + 6 + 30 + 44 = 82
  // 2; [1*3 + 2*4 + 3*11 + 4*12] = 3 + 8 + 33 + 48 = 92
  // 3: [ 14 + 25 + 312 + 413 = 4 + 10 + 36 + 52 = 102
  // 4: [ 15 + 26 + 315 + 416 = 5 + 12 + 39 + 56 = 112
  // 5: [ 16 + 27 + 318 + 419 = 6 + 14 + 42 + 60 = 122
  // 6: [ 17 + 28 + 321 + 422 = 7 + 16 + 45 + 64 = 132
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
