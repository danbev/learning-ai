#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <stdio.h>
#include <string.h>

int main(int argc, char **argv) {
  printf("GGML galloc (graph allocator) example\n");

  struct ggml_init_params params = {
    .mem_size   = 16*1024*1024,
    .mem_buffer = NULL,
    .no_alloc   = true,
  };
  struct ggml_context* ctx = ggml_init(params);

  struct ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 3);
  ggml_set_name(a, "a");
  float a_data[6] = { 1, 2, 3, 4, 5, 6};

  struct ggml_tensor* b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2);
  ggml_set_name(b, "b");
  float b_data[2] = { 2, 3};

  // Note that ggml_mul_mat() transposes the second matrix b.
  struct ggml_tensor* result = ggml_mul(ctx, a, b);
  ggml_set_name(result, "result");

  struct ggml_cgraph* c_graph = ggml_new_graph(ctx);
  ggml_build_forward_expand(c_graph, result);

  ggml_backend_t backend = ggml_backend_cpu_init();

  //ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
  //ggml_backend_tensor_set(a, a_data, 0, ggml_nbytes(a));  
  //ggml_backend_tensor_set(b, b_data, 0, ggml_nbytes(b));  
  // For the CPU backend this is pretty much just a memcpy:
  //memcpy((char *)b->data + 0, b_data, ggml_nbytes(b));

  ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
  ggml_gallocr_alloc_graph(galloc, c_graph);

  ggml_backend_cpu_set_n_threads(backend, 1);
  enum ggml_status st = ggml_backend_graph_compute(backend, c_graph);
  if (st != GGML_STATUS_SUCCESS) {
    printf("could not compute graph\n");
    return 1;
  }

  printf("result matrix %ldx%ld:\n", result->ne[0], result->ne[1]);

  float result_data[6];
  ggml_backend_tensor_get(result, result_data, 0, ggml_nbytes(result));
  for (int y = 0; y < result->ne[1]; y++) {
      for (int x = 0; x < result->ne[0]; x++) {
          printf("%.2f ", *(float *) ((char *) result->data + y * result->nb[1] + x * result->nb[0]));
       }
      printf("\n");
  }

  ggml_gallocr_free(galloc);
  ggml_free(ctx);
  //ggml_backend_buffer_free(buffer);
  ggml_backend_free(backend);
  return 0;
}
