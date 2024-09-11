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

  //view = ggml_view_1d(ctx, x, 5, (5-1) * ggml_type_size(x->type));
  view = ggml_view_1d(ctx, x, 0, (5-1) * ggml_type_size(x->type));

  ggml_print_objects(ctx);

  for (int i = 0; i < ggml_nelements(view); i++) {
    printf("view[%d]: %f\n", i, ggml_get_f32_1d(view, i));
  }

  // this is the stride to move to the next row.
  size_t nb1 = 8;
  size_t offset = 0;
  struct ggml_tensor* view_2d = ggml_view_2d(ctx, x, 2, 3, nb1, offset);
  printf("view_2d tensor rows: %ld\n", ggml_nrows(view_2d));
  printf("view_2d tensor elements: %ld\n", ggml_nelements(view_2d));

  printf("%f\n", ((float*) view_2d->data)[0]);
  printf("%f\n", ((float*) view_2d->data)[1]);
  printf("%f\n", ((float*) view_2d->data)[2]);
  printf("%f\n", ((float*) view_2d->data)[3]);
  printf("%f\n", ((float*) view_2d->data)[4]);
  printf("%f\n", ((float*) view_2d->data)[5]);

  printf("view_2d get using nd x:\n");
  printf("%f\n", ggml_get_f32_nd(view_2d, 0, 0, 0, 0));
  printf("%f\n", ggml_get_f32_nd(view_2d, 1, 0, 0, 0));
  printf("%f\n", ggml_get_f32_nd(view_2d, 2, 0, 0, 0));
  printf("%f\n", ggml_get_f32_nd(view_2d, 3, 0, 0, 0));
  printf("%f\n", ggml_get_f32_nd(view_2d, 4, 0, 0, 0));
  printf("%f\n", ggml_get_f32_nd(view_2d, 5, 0, 0, 0));

  printf("view_2d get using nd y:\n");
  printf("%f\n", ggml_get_f32_nd(view_2d, 0, 0, 0, 0));
  printf("%f\n", ggml_get_f32_nd(view_2d, 1, 0, 0, 0));
  printf("%f\n", ggml_get_f32_nd(view_2d, 0, 1, 0, 0));
  printf("%f\n", ggml_get_f32_nd(view_2d, 1, 1, 0, 0));
  printf("%f\n", ggml_get_f32_nd(view_2d, 0, 2, 0, 0));
  printf("%f\n", ggml_get_f32_nd(view_2d, 1, 2, 0, 0));

  printf("nb[0]=%ld\n", view_2d->nb[0]);
  printf("nb[1]=%ld\n", view_2d->nb[1]);
  printf("nb[2]=%ld\n", view_2d->nb[2]);
  printf("nb[3]=%ld\n", view_2d->nb[3]);

  printf("view_2d get using raw first dim only:\n");
  printf("%f\n", *(float*) ((char *) view_2d->data + 0 * view_2d->nb[0] + 0 * view_2d->nb[1] + 0 * view_2d->nb[2] + 0 * view_2d->nb[3]));
  printf("%f\n", *(float*) ((char *) view_2d->data + 1 * view_2d->nb[0] + 0 * view_2d->nb[1] + 0 * view_2d->nb[2] + 0 * view_2d->nb[3]));
  printf("%f\n", *(float*) ((char *) view_2d->data + 2 * view_2d->nb[0] + 0 * view_2d->nb[1] + 0 * view_2d->nb[2] + 0 * view_2d->nb[3]));
  printf("%f\n", *(float*) ((char *) view_2d->data + 3 * view_2d->nb[0] + 0 * view_2d->nb[1] + 0 * view_2d->nb[2] + 0 * view_2d->nb[3]));
  printf("%f\n", *(float*) ((char *) view_2d->data + 4 * view_2d->nb[0] + 0 * view_2d->nb[1] + 0 * view_2d->nb[2] + 0 * view_2d->nb[3]));
  printf("%f\n", *(float*) ((char *) view_2d->data + 5 * view_2d->nb[0] + 0 * view_2d->nb[1] + 0 * view_2d->nb[2] + 0 * view_2d->nb[3]));

  printf("view_2d get using raw2:\n");
  printf("%f\n", *(float*) ((char *) view_2d->data + 0 * view_2d->nb[0] + 0 * view_2d->nb[1] + 0 * view_2d->nb[2] + 0 * view_2d->nb[3]));
  printf("%f\n", *(float*) ((char *) view_2d->data + 1 * view_2d->nb[0] + 0 * view_2d->nb[1] + 0 * view_2d->nb[2] + 0 * view_2d->nb[3]));
  printf("%f\n", *(float*) ((char *) view_2d->data + 0 * view_2d->nb[0] + 1 * view_2d->nb[1] + 0 * view_2d->nb[2] + 0 * view_2d->nb[3]));
  printf("%f\n", *(float*) ((char *) view_2d->data + 1 * view_2d->nb[0] + 1 * view_2d->nb[1] + 0 * view_2d->nb[2] + 0 * view_2d->nb[3]));
  printf("%f\n", *(float*) ((char *) view_2d->data + 0 * view_2d->nb[0] + 2 * view_2d->nb[1] + 0 * view_2d->nb[2] + 0 * view_2d->nb[3]));
  printf("%f\n", *(float*) ((char *) view_2d->data + 1 * view_2d->nb[0] + 2 * view_2d->nb[1] + 0 * view_2d->nb[2] + 0 * view_2d->nb[3]));

  struct ggml_tensor* zero = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 0);
  struct ggml_tensor* zero_view = ggml_view_1d(ctx, zero, 0, 1024);
  printf("zero_view tensor rows: %ld\n", ggml_nrows(zero_view));
  printf("zero_view tensor elements: %ld\n", ggml_nelements(zero_view));
  struct ggml_tensor* zero_view2 = ggml_view_1d(ctx, zero, 0, 1024);
  struct ggml_tensor* result = ggml_cpy(ctx, zero_view, zero_view2);
  printf("result tensor rows: %ld\n", ggml_nrows(result));
  printf("result tensor elements: %ld\n", ggml_nelements(result));
  //ggml_set_f32_1d(zero, 0, 1); // this will cause a segfault.

  ggml_free(ctx);

  return 0;
}
