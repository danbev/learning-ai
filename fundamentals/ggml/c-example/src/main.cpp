#include <iostream>

#include "ggml/ggml.h"
#include "ggml/ggml-alloc.h"

int main()
{
  std::cout << "GGML Example" << std::endl;
  struct ggml_init_params params = {
    .mem_size   = 16*1024*1024,
    .mem_buffer = NULL,
  };
  struct ggml_context *ctx = ggml_init(params);
  ggml_free(ctx);
  return 0;
}
