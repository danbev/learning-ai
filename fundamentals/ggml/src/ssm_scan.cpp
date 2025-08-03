#include <stdio.h>

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

int main(int argc, char **argv) {
  printf("GGML ssm_scan example\n");

  struct ggml_init_params params = {
    .mem_size   = 16*1024*1024,
    .mem_buffer = NULL,
    .no_alloc   = true,
  };
  struct ggml_context* ctx = ggml_init(params);
  
  // Initialize backend
  ggml_backend_t backend = ggml_backend_cpu_init();
  
  // d_inner is the dimension of the inner layer (after the projection layer).
  int d_inner = 8;
  // seq_len is the length of the input sequence.
  int seq_len = 4;
  // d_state is the dimension of the state vector
  int d_state = 16;

  // s is the current state of the system.
  // 
  //             d_state
  //       0 [0  ...        15]
  //       1 [0  ...        15]
  //       2 [0  ...        15]
  //       3 [0  ...        15]     d_inner
  //       4 [0  ...        15]
  //       5 [0  ...        15]
  //       6 [0  ...        15]
  //       7 [0  ...        15]
  //
  struct ggml_tensor* s = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_state , d_inner);
  ggml_set_name(s, "s");
  ggml_set_zero(s);
  printf("s nelements: %lld:\n", ggml_nelements(s));
  /*
  for (int i = 0; i < ggml_nelements(s); i++) {
	printf("%.2f ", ggml_get_f32_1d(s, i));
  }
  printf("\n");
  */

  // x is the output of the input->projection->convolution->silu.
  //
  //           d_inner
  // token 0 [0  ...  7]
  // token 1 [0  ...  7]     seq_len
  // token 2 [0  ...  7]
  // token 3 [0  ...  7]
  //
  struct ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_inner, seq_len);
  ggml_set_name(x, "x");
  printf("x nelements: %lld:\n", ggml_nelements(x));
  printf("x ne[0]: %lld:\n", x->ne[0]);
  printf("x ne[1]: %lld:\n", x->ne[1]);
  printf("x ne[2]: %lld:\n", x->ne[2]);
  ggml_set_f32_nd(x, 0, 0, 0, 0, 1.0f);

  // dt is the delta and we have one delta value per token.
  //
  //           d_inner
  // token 0 [0  ...  7]
  // token 1 [0  ...  7]     seq_len
  // token 2 [0  ...  7]
  // token 3 [0  ...  7]
  //
  struct ggml_tensor* dt = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_inner, seq_len);
  ggml_set_name(dt, "delta");

  // A is the learned state transition matrix.
  //
  //              d_state
  //       0 [0  ...        15]
  //       1 [0  ...        15]
  //       2 [0  ...        15]
  //       3 [0  ...        15]     d_inner
  //       4 [0  ...        15]
  //       5 [0  ...        15]
  //       6 [0  ...        15]
  //       7 [0  ...        15]
  //
  struct ggml_tensor* A = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_state, d_inner);
  ggml_set_name(A, "A");

  // B is the dynamic (not here but in a real Mamba model) input transition matrix.
  //           d_state
  // token 0 [0  ...  7]
  // token 1 [0  ...  7]     seq_len
  // token 2 [0  ...  7]
  // token 3 [0  ...  7]
  //
  struct ggml_tensor* B = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_state, seq_len);
  ggml_set_name(B, "B");

  // C is the output transition matrix.
  //           d_state
  // token 0 [0  ...  7]
  // token 1 [0  ...  7]     seq_len
  // token 2 [0  ...  7]
  // token 3 [0  ...  7]
  //
  struct ggml_tensor* C = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_state, seq_len);
  ggml_set_name(C, "C");

  // ids is typically used for batch processing - create a simple ids tensor
  struct ggml_tensor* ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, seq_len);
  ggml_set_name(ids, "ids");

  // y is the result of the scan operation. Which is a one dimensional tensor 
  // with the number of elements of x plus the number of elements in s.
  //
  // [0    ...  31   ... 160]
  //
  struct ggml_tensor* y = ggml_ssm_scan(ctx, s, x, dt, A, B, C, ids);
  ggml_set_name(y, "y");

  struct ggml_cgraph* c_graph = ggml_new_graph(ctx);
  ggml_build_forward_expand(c_graph, y);

  // Allocate tensors using backend
  ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);

  // Set thread count
  ggml_backend_cpu_set_n_threads(backend, 1);
  
  // Compute graph using backend
  enum ggml_status st = ggml_backend_graph_compute(backend, c_graph);
  if (st != GGML_STATUS_SUCCESS) {
    printf("could not compute graph\n");
    return 1;
  }

  printf("y nelements: %lld:\n", ggml_nelements(y));
  printf("y ne[0]: %lld:\n", y->ne[0]);

  // Cleanup
  ggml_backend_buffer_free(buffer);
  ggml_backend_free(backend);
  ggml_free(ctx);
  return 0;
}

