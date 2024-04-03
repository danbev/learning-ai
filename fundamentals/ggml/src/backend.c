#include <stdio.h>

#include "ggml/ggml.h"
#include "ggml/ggml-alloc.h"
#include "ggml/ggml-backend.h"

int main(int argc, char **argv) {
  printf("GGML backend examples\n");

  struct ggml_init_params params = {
    .mem_size   = 16*1024*1024,
    .mem_buffer = NULL,
  };
  struct ggml_context* ctx = ggml_init(params);

  // The following will call ggml_backend_registry_init
  size_t count = ggml_backend_reg_get_count();
  printf("backend count: %ld\n", count);
  printf("backend name: %s\n", ggml_backend_reg_get_name(0));

  ggml_free(ctx);
  return 0;
}
