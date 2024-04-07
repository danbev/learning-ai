#include <stdio.h>

#include "ggml/ggml.h"
#include "ggml/ggml-alloc.h"
#include "ggml/ggml-backend.h"

int main(int argc, char **argv) {
  printf("GGML backend examples\n");

  struct ggml_init_params params = {
    .mem_size   = 16*1024*1024,
    .mem_buffer = NULL,
    .no_alloc = true,
  };
  struct ggml_context* ctx = ggml_init(params);
  struct ggml_tensor* x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10);
  printf("x backend type (0=CPU, 10=GPU): %d\n", x->backend);
  if (x->buffer == NULL) {
    printf("x backend buffer is NULL\n");
  } else {
    printf("x backend buffer: %s\n", ggml_backend_buffer_name(x->buffer));
  }

  // The following will call ggml_backend_registry_init
  size_t count = ggml_backend_reg_get_count();
  printf("backend count: %ld\n", count);
  for (size_t i = 0; i < count; i++) {
    printf("backend_%ld name: %s\n", i, ggml_backend_reg_get_name(i));
  }

  ggml_backend_t cuda_backend = ggml_backend_reg_init_backend_from_str("CUDA0");
  if (cuda_backend != NULL) {
      ggml_backend_buffer_t buffer = ggml_backend_alloc_buffer(cuda_backend, 10*4);
      ggml_backend_buffer_type_t buffer_type = ggml_backend_buffer_get_type(buffer);

      ggml_backend_buffer_t bb = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buffer_type);

      printf("x backend type (0=CPU, 10=GPU): %d\n", x->backend);
      printf("x backend buffer: %s\n", ggml_backend_buffer_name(x->buffer));

      static float data_array[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
      void* data = (void*) data_array;

      // The following will copy the data from the host to the device.
      ggml_backend_tensor_set(x, data, 0, 10);
      printf("x backend type (0=CPU, 10=GPU): %d\n", x->backend);
  }

  ggml_backend_free(cuda_backend);
  ggml_free(ctx);
  return 0;
}
