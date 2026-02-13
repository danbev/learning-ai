#include <stdio.h>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"

// This example demonstrates how to use ggml_solve_tri to solve a linear system
// with a lower triangular matrix.
//
// [ 2  0  0 ]  [x1]   [4 ]
// [ 3  4  0 ]  [x2] = [22]
// [ 1  2  5 ]  [x3]   [25]
//
// And we are solving for x = [x1, x2, x3].
// The operation is solved in the following steps:
// 1) Solve first row x1: 2 * x1 + 0 * x2 + 0 * x2 = 4
//              x1 = 4 / 2 = 2
//              x1 = 2                                       [2]
// 2) Solve second row x2: 3 * x1 + 4 * x2 + 0 * x2 = 22
//                         3 * 2  + 4 * x2 + 0 * x2 = 22
//                         6 + 4 * x2 = 22
//                         4 * x2 = 16
//                         x2 = 16 / 4
//                         x2 = 4                            [4]
// 3) Solve third row: x3: 1 * x1 + 2 * x2 + 5 * x3 = 25
//                         1 *  2 + 2 *  4 + 5 * x3 = 25
//                         2 + 8 + 5 * x3 = 25
//                         10 + 5 * x3 = 25
//                         5 * x3 = 15
//                         x3 = 15 / 5
//                         x3 = 3                            [3]
//
int main(int argc, char** argv) {
  printf("GGML ggml_solve_tri example\n");

  struct ggml_init_params params = {
    .mem_size   = 16*1024*1024,
    .mem_buffer = NULL,
    .no_alloc   = true,
  };
  struct ggml_context* ctx = ggml_init(params);

  struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 3);
  struct ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 3);

  // Use ggml_solve_tri to solve Ax = b
  // Parameters: left=true (solve Ax=B), lower=true (lower triangular), uni=false (not unit diagonal)
  struct ggml_tensor * x = ggml_solve_tri(ctx, a, b, true, true, false);

  ggml_backend_t backend = ggml_backend_cpu_init();
  ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);

  float a_data[] = {
    2.0f, 0.0f, 0.0f,
    3.0f, 4.0f, 0.0f,
    1.0f, 2.0f, 5.0f
  };
  ggml_backend_tensor_set(a, a_data, 0, sizeof(a_data));

  float b_data[] = {4.0f, 22.0f, 25.0f};
  ggml_backend_tensor_set(b, b_data, 0, sizeof(b_data));

  printf("Lower triangular matrix a (3x3):\n");
  float a_out[9];
  ggml_backend_tensor_get(a, a_out, 0, sizeof(a_out));
  for (int row = 0; row < 3; row++) {
    printf("  [");
    for (int col = 0; col < 3; col++) {
      printf(" %.1f", a_out[row * 3 + col]);
    }
    printf(" ]\n");
  }

  printf("\nVector b:\n");
  float b_out[3];
  ggml_backend_tensor_get(b, b_out, 0, sizeof(b_out));
  printf("  [ %.1f %.1f %.1f ]\n", b_out[0], b_out[1], b_out[2]);

  struct ggml_cgraph* gf = ggml_new_graph(ctx);
  ggml_build_forward_expand(gf, x);

  ggml_backend_graph_compute(backend, gf);

  printf("\nSolution x (solving ax = b):\n");
  float x_out[3];
  ggml_backend_tensor_get(x, x_out, 0, sizeof(x_out));
  printf("  [ %.1f %.1f %.1f ]\n", x_out[0], x_out[1], x_out[2]);
  printf("Expected: [ 2.0 4.0 3.0 ]\n");

  ggml_backend_buffer_free(buffer);
  ggml_backend_free(backend);
  ggml_free(ctx);

  return 0;
}
