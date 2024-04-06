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
  ggml_backend_t cpu_backend = ggml_backend_cpu_init();

  printf("backend name: %s\n", ggml_backend_name(cpu_backend));
  printf("backend alignment: %ld\n", ggml_backend_get_alignment(cpu_backend));
  printf("backend max_size: %ld\n", ggml_backend_get_max_size(cpu_backend));

  // The following will call ggml_backend_registry_init
  size_t count = ggml_backend_reg_get_count();
  printf("backend count: %ld\n", count);
  printf("backend name: %s\n", ggml_backend_reg_get_name(0));

  ggml_backend_buffer_t backend_buffer = ggml_backend_alloc_buffer(cpu_backend, 10*4);
  struct ggml_tensor* x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10);
  // Is optional and is not implemented for the CPU backend.
  //ggml_backend_buffer_init_tensor(backend_buffer, x);

  ggml_backend_tensor_set(x, x->data, 0, ggml_nbytes(x));

  ggml_backend_free(cpu_backend);
  ggml_free(ctx);
  return 0;
}
