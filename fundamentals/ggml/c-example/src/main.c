#include <stdio.h>

#include "ggml/ggml.h"
#include "ggml/ggml-alloc.h"

int main(int argc, char **argv) {
  printf("GGML Example\n");

  struct ggml_init_params params = {
    .mem_size   = 16*1024*1024,
    .mem_buffer = NULL,
  };
  struct ggml_context *ctx = ggml_init(params);
  printf("ctx mem size: %ld\n", ggml_get_mem_size(ctx));
  printf("ctx mem used: %ld\n", ggml_used_mem(ctx));
  ggml_free(ctx);
  return 0;
}
