#include <stdio.h>

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

int main(int argc, char **argv) {
  printf("GGML dup example\n");

  struct ggml_init_params params = {
    .mem_size   = 16*1024*1024,
    .mem_buffer = NULL,
    .no_alloc   = false,
  };

  struct ggml_context* ctx = ggml_init(params);
  struct ggml_tensor* x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10);
  ggml_set_name(x, "x");
  ggml_set_i32(x, 18);
  for (int i = 0; i < ggml_nelements(x); i++) {
    printf("x[%d]: %d\n", i, ggml_get_i32_1d(x, i));
  }

  struct ggml_tensor* dup = ggml_dup_tensor(ctx, x);
  // Only the tensor type and its dimensions are duplicated, that is a new
  // tensor is created with the same type and dimensions.
  printf("dup tensor type: %s\n", ggml_type_name(dup->type));
  printf("dup tensor dimensions: %d\n", ggml_n_dims(dup));
  printf("x tensor ne[0]: %ld\n", dup->ne[0]);
  printf("x tensor n3[1]: %ld\n", dup->ne[1]);
  printf("x tensor n3[2]: %ld\n", dup->ne[2]);
  printf("x tensor n3[3]: %ld\n", dup->ne[3]);

  // name is not duplicated
  printf("name is not duplicate dup tensor name: %s\n", dup->name);

  printf("And values are not duplicated either:\n");
  for (int i = 0; i < ggml_nelements(dup); i++) {
    printf("x[%d]: %d\n", i, ggml_get_i32_1d(dup, i));
  }

  ggml_free(ctx);
  return 0;
}
