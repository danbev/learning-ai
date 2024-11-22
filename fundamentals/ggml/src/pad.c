#include <stdio.h>

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

int main(int argc, char **argv) {
  printf("GGML pad example\n");

  struct ggml_init_params params = {
    .mem_size   = 16*1024*1024,
    .mem_buffer = NULL,
  };
  struct ggml_context* ctx = ggml_init(params);

  struct ggml_tensor* one = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3);
  ggml_set_f32_nd(one, 0, 0, 0, 0, 100);
  ggml_set_f32_nd(one, 1, 0, 0, 0, 200);
  ggml_set_f32_nd(one, 2, 0, 0, 0, 300);
  // So we start out with the following tensor:
  //    [100, 200, 300]
  printf("Only has %ld elements\n", ggml_nelements(one));
  printf("\n");

  struct ggml_tensor* result = ggml_pad(ctx, one, 2, 0, 1, 0);
  // And padding will create a new tensor:
  // z_0
  //    [100, 200, 300, 0, 0]
  // z_1
  //    [  0,   0,   0, 0, 0]
  ggml_set_name(result, "result");

  struct ggml_cgraph* c_graph = ggml_new_graph(ctx);
  ggml_build_forward_expand(c_graph, result);
  enum ggml_status st = ggml_graph_compute_with_ctx(ctx, c_graph, 1);
  if (st != GGML_STATUS_SUCCESS) {
    printf("could not compute graph\n");
    return 1;
  }
  printf("result: dims: %d\n", ggml_n_dims(result));
  printf("result: ne[0]: %ld\n", result->ne[0]);
  printf("result: ne[1]: %ld\n", result->ne[1]);
  printf("result: ne[2]: %ld\n", result->ne[2]);
  printf("result: ne[3]: %ld\n", result->ne[3]);
  printf("result: elements: %ld\n", ggml_nelements(result));

  printf("result: z_0 0: %f\n", ggml_get_f32_nd(result, 0, 0, 0, 0));
  printf("result: z_0 1: %f\n", ggml_get_f32_nd(result, 1, 0, 0, 0));
  printf("result: z_0 2: %f\n", ggml_get_f32_nd(result, 2, 0, 0, 0));
  printf("result: z_0 3: %f\n", ggml_get_f32_nd(result, 3, 0, 0, 0));
  printf("result: z_0 4: %f\n", ggml_get_f32_nd(result, 4, 0, 0, 0));
  printf("\n");

  printf("result: z_1 0: %f\n", ggml_get_f32_nd(result, 0, 0, 1, 0));
  printf("result: z_1 1: %f\n", ggml_get_f32_nd(result, 1, 0, 1, 0));
  printf("result: z_1 2: %f\n", ggml_get_f32_nd(result, 2, 0, 1, 0));
  printf("result: z_1 3: %f\n", ggml_get_f32_nd(result, 3, 0, 1, 0));
  printf("result: z_1 4: %f\n", ggml_get_f32_nd(result, 4, 0, 1, 0));

  ggml_free(ctx);
  return 0;
}
