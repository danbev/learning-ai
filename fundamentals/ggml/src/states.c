#include <stdio.h>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

int main(int argc, char **argv) {
  printf("GGML get_rows (states) example\n");

  struct ggml_init_params params = {
    .mem_size   = 16*1024*1024,
    .mem_buffer = NULL,
  };
  struct ggml_context* ctx = ggml_init(params);

  struct ggml_tensor* states = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, 2, 3, 1);
  printf("states n_dims: %d\n", ggml_n_dims(states));
  printf("states n_elements: %ld\n", ggml_nelements(states));
  printf("states before get_rows\n");
  ggml_set_i32_nd(states, 0, 0, 0, 0, 1);
  ggml_set_i32_nd(states, 1, 0, 0, 0, 2);
  ggml_set_i32_nd(states, 0, 1, 0, 0, 3);
  ggml_set_i32_nd(states, 1, 1, 0, 0, 4);
  ggml_set_i32_nd(states, 0, 2, 0, 0, 5);
  ggml_set_i32_nd(states, 1, 2, 0, 0, 6);
  for (int y = 0; y < states->ne[1]; y++) {
      printf("states %d: ", y);
      for (int x = 0; x < states->ne[0]; x++) {
          printf("%d ", *(int *) ((char *) states->data + y * states->nb[1] + x * states->nb[0]));
       }
      printf("\n");
  }
  printf("\n");

  struct ggml_tensor* state_copy = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 3);
  ggml_set_i32_1d(state_copy, 0, 0);
  ggml_set_i32_1d(state_copy, 1, 1);
  states = ggml_get_rows(ctx, states, state_copy);

  struct ggml_cgraph* c_graph = ggml_new_graph(ctx);
  ggml_build_forward_expand(c_graph, states);
  ggml_graph_compute_with_ctx(ctx, c_graph, 1);

  printf("states after get_rows\n");
  printf("states elements: %ld\n", ggml_nelements(states));
  for (int y = 0; y < states->ne[1]; y++) {
      printf("states %d: ", y);
      for (int x = 0; x < states->ne[0]; x++) {
          printf("%d ", *(int *) ((char *) states->data + y * states->nb[1] + x * states->nb[0]));
       }
      printf("\n");
  }
  printf("\n");
  printf("%d\n", ggml_get_i32_nd(states, 0, 0, 0, 0));
  printf("%d\n", ggml_get_i32_nd(states, 1, 0, 0, 0));
  printf("%d\n", ggml_get_i32_nd(states, 2, 0, 0, 0));
  printf("%d\n", ggml_get_i32_nd(states, 3, 0, 0, 0));
  // The following will print out 1
  printf("%d\n", ggml_get_i32_nd(states, 4, 0, 0, 0));
  printf("%d\n", ggml_get_i32_nd(states, 5, 0, 0, 0));

  ggml_free(ctx);
  return 0;
}
