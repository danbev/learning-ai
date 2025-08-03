#include <stdio.h>
#include <string.h>
#include <math.h>

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

/*
 * This example is intended to get a feel/understanding for how the attention mask
 * used with the QK attention scores work in Llama.cpp's attention softmax.
 */
int main(int argc, char **argv) {
  printf("GGML llama attention softmax example\n");

  struct ggml_init_params params = {
    .mem_size   = 16*1024*1024,
    .mem_buffer = NULL,
  };
  struct ggml_context* ctx = ggml_init(params);

  struct ggml_tensor* logits = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 32, 1, 1);
  ggml_set_name(logits, "logits");

  float tensor_data[32] = {
      0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
      10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0,
      20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0,
      30.0, 31.0
  };
  memcpy((char *)logits->data, tensor_data, ggml_nbytes(logits));
  for (int i = 0; i < ggml_nelements(logits); i++) {
	float value = *(float *) ((char *) logits->data + i * logits->nb[0]); 
	printf("%.4f ", value);
  }
  printf("\n");

  struct ggml_tensor* mask = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 32);
  ggml_set_name(mask, "mask");
  float mask_data[32] = {
      -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, 0.0f, 0.0f, 0.0f, 0.0f,
      0.0f     , 0.0f     , 0.0f     , 0.0f     , -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY,
      -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY,
      -INFINITY, -INFINITY,
  };
  memcpy((char *)mask->data, mask_data, ggml_nbytes(mask));
  for (int i = 0; i < ggml_nelements(mask); i++) {
      float value = *(float *) ((char *) mask->data + i * mask->nb[0]);
      printf("%.4f ", value);
  }
  printf("\n");

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
