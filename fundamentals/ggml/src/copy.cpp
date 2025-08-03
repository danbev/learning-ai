#include <stdio.h>

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

int main(int argc, char **argv) {
  printf("GGML copy examples\n");

  struct ggml_init_params params = {
    .mem_size   = 16*1024*1024,
    .mem_buffer = NULL,
  };
  struct ggml_context* ctx = ggml_init(params);

  struct ggml_tensor* a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5);
  ggml_set_name(a, "a");
  ggml_set_f32_1d(a, 0, 1);
  ggml_set_f32_1d(a, 1, 2);
  ggml_set_f32_1d(a, 2, 3);
  ggml_set_f32_1d(a, 3, 4);
  ggml_set_f32_1d(a, 4, 5);

  printf("a tensor:\n");
  for (int i = 0; i < ggml_nelements(a); i++) {
    printf("a[%d]: %f\n", i, ggml_get_f32_1d(a, i));
  }

  struct ggml_tensor* b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5);

  printf("b tensor:\n");
  for (int i = 0; i < ggml_nelements(b); i++) {
    printf("b[%d]: %f\n", i, ggml_get_f32_1d(b, i));
  }

  struct ggml_tensor* result = ggml_cpy(ctx, a, b);

  struct ggml_cgraph* c_graph = ggml_new_graph(ctx);
  ggml_build_forward_expand(c_graph, result);
  //ggml_build_forward_expand(c_graph, c);
  int n_threads = 4;
  enum ggml_status st = ggml_graph_compute_with_ctx(ctx, c_graph, n_threads);
  if (st != GGML_STATUS_SUCCESS) {
    printf("could not compute graph\n");
    return 1;
  }

  printf("copyied tensor a to b:\n");
  for (int i = 0; i < ggml_nelements(b); i++) {
    printf("b[%d]: %f\n", i, ggml_get_f32_1d(b, i));
  }

  struct ggml_tensor* c = ggml_view_1d(ctx, b, 2, 0);
  printf("c (view of b):\n");
  for (int i = 0; i < ggml_nelements(c); i++) {
    printf("c[%d]: %f\n", i, ggml_get_f32_1d(c, i));
  }

  struct ggml_tensor* d = ggml_view_1d(ctx, b, 0, 1000);
  printf("d (view of b): n_elements: %lld\n", ggml_nelements(d));
  for (int i = 0; i < ggml_nelements(d); i++) {
    printf("d[%d]: %f\n", i, ggml_get_f32_1d(d, i));
  }

  struct ggml_tensor* empty = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 0);
  printf("empty: n_elements: %lld\n", ggml_nelements(empty));

  struct ggml_tensor* ev1 = ggml_view_1d(ctx, empty, 0, 1000);
  struct ggml_tensor* ev2 = ggml_view_1d(ctx, empty, 0, 1000);

  struct ggml_tensor* r = ggml_cpy(ctx, ev1, ev2);
  printf("result: n_elements: %lld\n", ggml_nelements(r));

  ggml_free(ctx);
  return 0;
}
