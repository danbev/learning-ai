#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <stdio.h>

int main(int argc, char **argv) {
  printf("GGML galloc (graph allocator) example\n");

  struct ggml_init_params params = {
    .mem_size   = 16*1024*1024,
    .mem_buffer = NULL,
  };
  struct ggml_context* ctx = ggml_init(params);

  struct ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 3);
  ggml_set_name(a, "a");
  ggml_set_i32_nd(a, 0, 0, 0, 0, 1);
  ggml_set_i32_nd(a, 1, 0, 0, 0, 2);
  ggml_set_i32_nd(a, 0, 1, 0, 0, 3);
  ggml_set_i32_nd(a, 1, 1, 0, 0, 4);
  ggml_set_i32_nd(a, 0, 2, 0, 0, 5);
  ggml_set_i32_nd(a, 1, 2, 0, 0, 6);

  printf("matrix a %ldx%ld:\n", a->ne[0], a->ne[1]);
  for (int y = 0; y < a->ne[1]; y++) {
      for (int x = 0; x < a->ne[0]; x++) {
          printf("%.2f ", *(float *) ((char *) a->data + y * a->nb[1] + x * a->nb[0]));
       }
      printf("\n");
  }
  printf("\n");

  struct ggml_tensor* b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2);
  ggml_set_name(b, "b");
  ggml_set_i32_nd(b, 0, 0, 0, 0, 2);
  ggml_set_i32_nd(b, 1, 0, 0, 0, 4);

  // Note that ggml_mul_mat() transposes the second matrix b.
  struct ggml_tensor* result = ggml_mul(ctx, a, b);
  ggml_set_name(result, "result");

  struct ggml_cgraph* c_graph = ggml_new_graph(ctx);
  ggml_build_forward_expand(c_graph, result);

  ggml_backend_t cpu_backend = ggml_backend_reg_init_backend_from_str("CPU");
  
  ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(cpu_backend));
  ggml_gallocr_alloc_graph(galloc, c_graph);

  enum ggml_status st = ggml_backend_graph_compute(cpu_backend, c_graph);
  if (st != GGML_STATUS_SUCCESS) {
    printf("could not compute graph\n");
    return 1;
  }

  printf("result matrix %ldx%ld:\n", result->ne[0], result->ne[1]);
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
