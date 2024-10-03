#include <stdio.h>
#include <string.h>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

int main(int argc, char **argv) {
  printf("GGML im2col (image to column) example\n\n");

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
  int dil_x = 1;
  int dil_y = 1;
  bool is_2d = true;
  struct ggml_tensor* result = ggml_im2col(ctx, a, b, stride_x, stride_y, pad_x, pad_y, dil_x, dil_y, is_2d, GGML_TYPE_F32);

  struct ggml_cgraph* c_graph = ggml_new_graph(ctx);
  ggml_build_forward_expand(c_graph, result);
  ggml_graph_compute_with_ctx(ctx, c_graph, 1);

  printf("result dims: %d\n", ggml_n_dims(result));
  printf("result ne[0]: %ld\n", result->ne[0]);
  printf("result ne[1]: %ld\n", result->ne[1]);
  // The first row of the result tensor will be the first patch:
  //
  // a (convolution kernel)
  // 1.00 2.00
  // 3.00 4.00
  // 
  // b (data):
  // 1.00 2.00 3.00 4.00 5.00 6.00 7.00 8.00
  // 9.00 10.00 11.00 12.00 13.00 14.00 15.00 16.00
  //
  // First row:
  // 1.00 2.00 9.00 10.00
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
