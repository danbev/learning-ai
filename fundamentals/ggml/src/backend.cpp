#include <stdio.h>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

void print_backend_info(ggml_backend_buffer_t buffer, struct ggml_context* ctx) {
      printf("------- backend info -------\n");
      printf("buffer name: %s\n", ggml_backend_buffer_name(buffer));
      printf("buffer size: %ld\n", ggml_backend_buffer_get_size(buffer));
      printf("buffer alignment: %ld\n", ggml_backend_buffer_get_alignment(buffer));
      printf("buffer max size: %ld\n", ggml_backend_buffer_get_max_size(buffer));
      printf("buffer is host: %d\n", ggml_backend_buffer_is_host(buffer));

      ggml_backend_buffer_type_t buffer_type = ggml_backend_buffer_get_type(buffer);
      printf("buffer type name: %s\n", ggml_backend_buft_name(buffer_type));
      printf("buffer type alignment: %ld\n", ggml_backend_buft_get_alignment(buffer_type));
      printf("buffer type max size: %ld\n", ggml_backend_buft_get_max_size(buffer_type));
      printf("buffer type is host: %d\n", ggml_backend_buft_is_host(buffer_type));
}

int main(int argc, char **argv) {
  printf("GGML backend examples\n");

  struct ggml_init_params params = {
    .mem_size   = 16*1024*1024,
    .mem_buffer = NULL,
    .no_alloc = true,
  };
  struct ggml_context* ctx = ggml_init(params);
  struct ggml_tensor* x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10);
  ggml_set_name(x, "x");
  if (x->buffer == NULL) {
    printf("x backend buffer is NULL\n");
  } else {
    printf("x backend buffer: %s\n", ggml_backend_buffer_name(x->buffer));
  }

  // The following will call ggml_backend_registry_init
  size_t count = ggml_backend_reg_count();
  printf("Number of backends registered: %ld\n", count);
  for (size_t i = 0; i < count; i++) {
    printf("backend %ld name: %s\n", i, ggml_backend_dev_name(ggml_backend_dev_get(i)));
  }

  ggml_backend_t cpu_backend = ggml_backend_init_by_name("CPU", NULL);
  if (cpu_backend != NULL) {
      ggml_backend_buffer_t buffer = ggml_backend_alloc_buffer(cpu_backend, 10*4);
      print_backend_info(buffer, ctx);
      ggml_backend_free(cpu_backend);
  }

  ggml_backend_t cuda_backend = ggml_backend_init_by_name("CUDA0", NULL);
  if (cuda_backend != NULL) {
      ggml_backend_buffer_t buffer = ggml_backend_alloc_buffer(cuda_backend, 10*4);
      print_backend_info(buffer, ctx);
      if (x->buffer != NULL) {
          printf("x backend buffer: %s\n", ggml_backend_buffer_name(x->buffer));
      }

      ggml_backend_buffer_t t = ggml_backend_alloc_ctx_tensors(ctx, cuda_backend);

      static float data_array[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
      void* data = (void*) data_array;

      // The following will copy the data from the host to the device.
      ggml_backend_tensor_set(x, data, 0, 10);
      ggml_backend_free(cuda_backend);
  }

  ggml_free(ctx);
  return 0;
}
