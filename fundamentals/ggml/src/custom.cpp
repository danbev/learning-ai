#include <stdio.h>

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

void custom1_fun(struct ggml_tensor* dst ,
                 const struct ggml_tensor* a,
		 int ith,
		 int nth,
		 void* userdata) {
    char* user_data = (char*) userdata;
    printf("[custom1_fun] thread index: %d\n", ith);
    printf("[custom1_fun] number of threads: %d\n", nth);
    printf("[custom1_fun] user_data: %s\n", user_data);
}

/*
 * This example demonstrates how to use the custom1 function.
 */
int main(int argc, char **argv) {
  printf("GGML custom example\n");

  struct ggml_init_params params = {
    .mem_size   = 16*1024*1024,
    .mem_buffer = NULL,
  };
  struct ggml_context* ctx = ggml_init(params);

  struct ggml_tensor* a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5);
  char* user_data = "some user provided data";
  struct ggml_tensor* custom = ggml_map_custom1(ctx, a, custom1_fun, 2, (void*) user_data); 

  struct ggml_cgraph* c_graph = ggml_new_graph(ctx);
  ggml_build_forward_expand(c_graph, custom);

  int n_threads = 2;
  enum ggml_status st = ggml_graph_compute_with_ctx(ctx, c_graph, n_threads);
  if (st != GGML_STATUS_SUCCESS) {
    printf("could not compute graph\n");
    return 1;
  }

  ggml_free(ctx);
  return 0;
}
