#include <stdio.h>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

int main(int argc, char **argv) {
  printf("GGML cont tensor examples\n");

  struct ggml_init_params params = {
    .mem_size   = 16*1024*1024,
    .mem_buffer = NULL,
  };
  struct ggml_context* ctx = ggml_init(params);

  struct ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 3);
  ggml_set_name(a, "a");
  printf("a elements: %ld\n", ggml_nelements(a));
  printf("a ne[0]: %ld\n", a->ne[0]);
  printf("a ne[1]: %ld\n", a->ne[1]);
  printf("a ne[2]: %ld\n", a->ne[2]);
  printf("a ne[3]: %ld\n", a->ne[3]);
  printf("a is_contiguous: %d\n", ggml_is_contiguous(a));

  // the following will make the 2d tensor contiguous.
  struct ggml_tensor* c = ggml_cont_2d(ctx, a, 6, 1);
  printf("c elements: %ld\n", ggml_nelements(a));
  printf("c.name: '%s'\n", c->name);
  printf("c ne[0]: %ld\n", c->ne[0]);
  printf("c ne[1]: %ld\n", c->ne[1]);
  printf("c ne[2]: %ld\n", c->ne[2]);
  printf("c ne[3]: %ld\n", c->ne[3]);
  printf("c is_contiguous: %d\n", ggml_is_contiguous(c));

  ggml_free(ctx);
  return 0;
}
