#include <stdio.h>

#include "ggml/ggml.h"
#include "ggml/ggml-alloc.h"
#include "ggml/ggml-backend.h"

int main(int argc, char **argv) {
  printf("GGML reshape tensor examples\n");

  struct ggml_init_params params = {
    .mem_size   = 16*1024*1024,
    .mem_buffer = NULL,
  };
  struct ggml_context* ctx = ggml_init(params);

  struct ggml_tensor* org = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 5, 2);
  printf("org tensor type: %s\n", ggml_type_name(org->type));
  printf("org tensor dimensions: %d\n", ggml_n_dims(org));
  printf("org.ne[0]: %ld\n", org->ne[0]);
  printf("org.ne[1]: %ld\n", org->ne[1]);

  struct ggml_tensor* transposed = ggml_transpose(ctx, org);
  printf("transposed tensor type: %s\n", ggml_type_name(transposed->type));
  printf("transposed tensor dimensions: %d\n", ggml_n_dims(transposed));
  printf("transposed.ne[0]: %ld\n", transposed->ne[0]);
  printf("transposed.ne[1]: %ld\n", transposed->ne[1]);

  ggml_free(ctx);
  return 0;
}
