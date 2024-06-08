#include <stdio.h>

#include "ggml.h"
#include "ggml-quants.h"

#ifdef GGML_COMMON_DECL_C
#warning "[danbev] GGML_COMMON_DECL_C is defined"
#endif

#ifdef GGML_COMMON_DECL_CUDA
#warning "[danbev] GGML_COMMON_DECL_CUDA is defined"
#endif

int main(int argc, char **argv) {
  printf("GGML Quantization examples\n");

  struct ggml_init_params params = {
    .mem_size   = 16*1024*1024,
    .mem_buffer = NULL,
  };
  struct ggml_context* ctx = ggml_init(params);



  ggml_free(ctx);
  return 0;
}
