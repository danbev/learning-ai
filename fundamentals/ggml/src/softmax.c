#include <stdio.h>
#include <string.h>
#include <math.h>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

int main(int argc, char **argv) {
  printf("GGML softmax example\n");

  struct ggml_init_params params = {
    .mem_size   = 16*1024*1024,
    .mem_buffer = NULL,
  };
  struct ggml_context* ctx = ggml_init(params);

  struct ggml_tensor* logits = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4, 1, 1);
  ggml_set_name(logits, "logits");

  float tensor_data[4] = { 6, 7, 10, 9};
  memcpy((char *)logits->data, tensor_data, ggml_nbytes(logits));

  struct ggml_tensor* mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 1);
                                
  ggml_set_name(mask, "mask");
  float mask_data[4] = { 0, 0, -INFINITY, 0};
  memcpy((char *)mask->data, mask_data, ggml_nbytes(mask));

  struct ggml_tensor* result = ggml_soft_max_ext(ctx, logits, mask, 1.0f, 0.0f);
  ggml_set_name(result, "result");

  struct ggml_cgraph* c_graph = ggml_new_graph(ctx);
  ggml_build_forward_expand(c_graph, result);
  int n_threads = 1;
  enum ggml_status st = ggml_graph_compute_with_ctx(ctx, c_graph, n_threads);
  if (st != GGML_STATUS_SUCCESS) {
    printf("could not compute graph\n");
    return 1;
  }

  printf("result tensor type: %s\n", ggml_type_name(result->type));
  printf("result dim: %d\n", ggml_n_dims(result));
  printf("result dim[0]: %ld\n", result->ne[0]);
  float sum = 0.0f;
  for (int i = 0; i < ggml_nelements(result); i++) {
    float value = *(float *) ((char *) result->data + i * result->nb[0]); 
	printf("%.4f ", value);
    sum += value;
  }
  printf("\nsum: %.4f\n", sum);

  ggml_free(ctx);
  return 0;
}
