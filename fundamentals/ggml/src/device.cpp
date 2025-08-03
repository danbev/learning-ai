#include <stdio.h>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

void print_backend_info(ggml_backend_buffer_t buffer, struct ggml_context* ctx) {
      printf("------- device info -------\n");
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
  printf("GGML device examples\n");
  size_t device_count = ggml_backend_dev_count();
  printf("device count: %ld\n", device_count);

  ggml_backend_dev_t device = ggml_backend_dev_get(0);
  printf("device name: %s\n", ggml_backend_dev_name(device));
  printf("device description: %s\n", ggml_backend_dev_description(device));

  enum ggml_backend_dev_type type = ggml_backend_dev_type(device);
  ggml_backend_t backend = ggml_backend_init_by_type(type, NULL);
  printf("backend name: %s\n", ggml_backend_name(backend));
  if (type == GGML_BACKEND_DEVICE_TYPE_CPU) {
      printf("backend type: GGML_BACKEND_DEVICE_TYPE_CPU\n");
  }

  ggml_backend_buffer_type_t buf_type = ggml_backend_get_default_buffer_type(backend);
  printf("buffer type name: %s\n", ggml_backend_buft_name(buf_type));

  ggml_backend_buffer_t buffer =  ggml_backend_buft_alloc_buffer(buf_type, 100);
  printf("buffer name: %s\n", ggml_backend_buffer_name(buffer));

  return 0;
}
