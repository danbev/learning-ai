#include <stdio.h>

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

int main(int argc, char **argv) {
  printf("GGML permute tensor examples\n");

  struct ggml_init_params params = {
    .mem_size   = (ggml_tensor_overhead() * 2) + (sizeof(float) * 3 * 2 * 4 * 1),
    .mem_buffer = NULL,
  };
  struct ggml_context* ctx = ggml_init(params);

  struct ggml_tensor* a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 3, 2, 4, 1);
  ggml_set_name(a, "a");

  printf("a ne[0]: %ld\n", a->ne[0]);
  printf("a ne[1]: %ld\n", a->ne[1]);
  printf("a ne[2]: %ld\n", a->ne[2]);
  printf("a ne[3]: %ld\n", a->ne[3]);

  // A permuation is similar to a transpose, but it is a generalization of the
  // transpose operation. The transpose operation is a special case of the
  // permute operation.
  // Ask: where do I want the the dim d of a to go in p.
  struct ggml_tensor* p = ggml_permute(ctx, a, 3, 0, 1, 2);
  // I want dim 0 of a, 3, to to in dim 3 of p:  [_, _, _, 3]
  // I want dim 1 of a, 2, to go in dim 0 of p:  [2, _, _, 3]
  // I want dim 2 or a, 4, to go in dim 1 of p:  [2, 4, _, 3]
  // I want dim 3 of a, 1, to go in dim 2 of p:  [2, 4, 1, 3]

  printf("Permuted:\n"); 
  printf("p ne[0]: %ld\n", p->ne[0]);
  printf("p ne[1]: %ld\n", p->ne[1]);
  printf("p ne[2]: %ld\n", p->ne[2]);
  printf("p ne[3]: %ld\n", p->ne[3]);

  ggml_free(ctx);
  return 0;
}
