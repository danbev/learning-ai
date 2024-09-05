#include <stdio.h>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

int main(int argc, char **argv) {
  printf("GGML view examples\n");

  struct ggml_init_params params = {
    .mem_size   = 16*1024*1024,
    .mem_buffer = NULL,
  };
  struct ggml_context* ctx = ggml_init(params);

  struct ggml_tensor* x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10);
  ggml_set_name(x, "x");
  ggml_set_f32_1d(x, 0, 1);
  ggml_set_f32_1d(x, 1, 2);
  ggml_set_f32_1d(x, 2, 3);
  ggml_set_f32_1d(x, 3, 4);
  ggml_set_f32_1d(x, 4, 5);
  ggml_set_f32_1d(x, 5, 6);
  ggml_set_f32_1d(x, 6, 7);
  ggml_set_f32_1d(x, 7, 8);
  ggml_set_f32_1d(x, 8, 9);
  ggml_set_f32_1d(x, 9, 10);

  for (int i = 0; i < ggml_nelements(x); i++) {
    printf("x[%d]: %f\n", i, ggml_get_f32_1d(x, i));
  }

  printf("x tensor dimensions: %d\n", ggml_n_dims(x));
  printf("x tensor elements: %ld\n", ggml_nelements(x));
  printf("x tensor type size: %ld\n", ggml_type_size(x->type));
  printf("x.ne: %ld\n", x->ne[0]);

  struct ggml_tensor* view = ggml_view_1d(ctx, x, 5, 0 * ggml_type_size(x->type));
  printf("view tensor dimensions: %d\n", ggml_n_dims(view));
  printf("view tensor elements: %ld\n", ggml_nelements(view));
  printf("view.ne: %ld\n", view->ne[0]);

  for (int i = 0; i < ggml_nelements(view); i++) {
    printf("view[%d]: %f\n", i, ggml_get_f32_1d(view, i));
  }

  view = ggml_view_1d(ctx, x, 5, (5-1) * ggml_type_size(x->type));

  ggml_print_objects(ctx);

  for (int i = 0; i < ggml_nelements(view); i++) {
    printf("view[%d]: %f\n", i, ggml_get_f32_1d(view, i));
  }

  ggml_free(ctx);
  return 0;
}
