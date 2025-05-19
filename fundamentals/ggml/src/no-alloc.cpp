#include "ggml.h"

#include <stdio.h>

int main(int argc, char **argv) {
  printf("GGML no_alloc example\n");

  printf("no_alloc=false\n");
  struct ggml_init_params params = {
    .mem_size   = 1024,
    .mem_buffer = NULL,
    .no_alloc   = false,
  };
  struct ggml_context* no_alloc_ctx = ggml_init(params);
  printf("no_alloc: %s\n", ggml_get_no_alloc(no_alloc_ctx) ? "true" : "false");

  struct ggml_tensor* a = ggml_new_tensor_1d(no_alloc_ctx, GGML_TYPE_F32, 1);
  ggml_set_name(a, "a");
  printf("a: n_elements: %ld\n", ggml_nelements(a));

  ggml_free(no_alloc_ctx);

  printf("no_alloc=true\n");
  struct ggml_init_params alloc_params = {
    .mem_size   = 1024,
    .mem_buffer = NULL,
    .no_alloc   = true,
  };
  struct ggml_context* alloc_ctx = ggml_init(alloc_params);
  printf("no_alloc: %s\n", ggml_get_no_alloc(alloc_ctx) ? "true" : "false");

  struct ggml_tensor* b = ggml_new_tensor_1d(alloc_ctx, GGML_TYPE_F32, 1);
  ggml_set_name(b, "b");
  printf("b: n_elements: %ld\n", ggml_nelements(b));
  // The following will cause a segfault
  //ggml_set_f32_1d(b, 0, 1.0f);

  ggml_free(alloc_ctx);

  printf("stack based mem_buffer\n");
  char stack_buffer[1024];
  struct ggml_init_params sb_params = {
    .mem_size   = 1024,
    .mem_buffer = stack_buffer,
    .no_alloc   = true,
  };
  struct ggml_context* sb_ctx = ggml_init(sb_params);
  printf("no_alloc: %s\n", ggml_get_no_alloc(sb_ctx) ? "true" : "false");

  struct ggml_tensor* c = ggml_new_tensor_1d(sb_ctx, GGML_TYPE_F32, 1);
  ggml_set_name(c, "c");
  printf("c: n_elements: %ld\n", ggml_nelements(c));

  ggml_free(sb_ctx);

  return 0;
}
