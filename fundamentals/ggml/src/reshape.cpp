#include <stdio.h>

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

int main(int argc, char **argv) {
  printf("GGML reshape tensor examples\n");

  struct ggml_init_params params = {
    .mem_size   = 16*1024*1024,
    .mem_buffer = NULL,
  };
  struct ggml_context* ctx = ggml_init(params);

  struct ggml_tensor* x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10);
  printf("x tensor dimensions: %d\n", ggml_n_dims(x));
  printf("x tensor elements: %ld\n", ggml_nelements(x));

  // Reshaping a 1d tensor to a 1d tensor does no really do much except copy
  // the tensor and set the name I think. But this could be used to reshaped
  // a 2d tensor to a 1d tensor for example.
  struct ggml_tensor* reshaped_1d = ggml_reshape_1d(ctx, x, 10);
  printf("reshaped_1d tensor type: %s\n", ggml_type_name(reshaped_1d->type));
  printf("reshaped_1d tensor dimensions: %d\n", ggml_n_dims(reshaped_1d));
  printf("reshaped_1d tensor elements: %ld\n", ggml_nelements(reshaped_1d));
  printf("reshaped_1d tensor name: %s\n", reshaped_1d->name);
  //
  // The following will reshape a 1d tensor to a 2d tensor.
  struct ggml_tensor* reshaped_2d = ggml_reshape_2d(ctx, x, 5, 2);
  printf("reshaped_2d tensor type: %s\n", ggml_type_name(reshaped_2d->type));
  printf("reshaped_2d tensor dimensions: %d\n", ggml_n_dims(reshaped_2d));
  printf("reshaped_2d tensor elements: %ld\n", ggml_nelements(reshaped_2d));
  printf("reshaped_2d tensor name: %s\n", reshaped_2d->name);
  for (int i = 0; i < reshaped_2d->ne[1]; i++) {
      printf("row: %i ", i);
      for (int j = 0; j < reshaped_2d->ne[0]; j++) {
          printf("%f ", ggml_get_f32_1d(reshaped_2d, i));
      }
      printf("\n");
  }


  ggml_free(ctx);
  return 0;
}
