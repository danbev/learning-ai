#include <stdio.h>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

int main(int argc, char **argv) {
  printf("GGML ssm_scan example\n");

  struct ggml_init_params params = {
    .mem_size   = 16*1024*1024,
    .mem_buffer = NULL,
  };
  struct ggml_context* ctx = ggml_init(params);
  // d_inner is the dimension of the inner layer (after the projection layer).
  int d_inner = 8;
  // seq_len is the length of the input sequence.
  int seq_len = 4;
  // d_state is the dimension of the state vector
  int d_state = 16;

  // s is the output of the input->projection->convolution->silu and is what will be scanned.
  //           d_state
  //       0 [0  ...  7]
  //       1 [0  ...  7]     d_inner
  //       2 [0  ...  7]
  //       3 [0  ...  7]
  //       4 [0  ...  7]
  //       5 [0  ...  7]
  //       6 [0  ...  7]
  //       7 [0  ...  7]
  struct ggml_tensor* s = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_state , d_inner);
  ggml_set_name(s, "s");
  ggml_set_zero(s);
  printf("s nelements: %lld:\n", ggml_nelements(s));
  for (int i = 0; i < ggml_nelements(s); i++) {
	printf("%.2f ", ggml_get_f32_1d(s, i));
  }
  printf("\n");

  // x is the token embeddings for the input sequence which consists of 4 tokens
  // of dimension 8.
  //           d_inner
  // token 0 [0  ...  7]
  // token 1 [0  ...  7]     seq_len
  // token 2 [0  ...  7]
  // token 3 [0  ...  7]
  struct ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_inner, seq_len);
  ggml_set_name(x, "x");
  ggml_set_f32_nd(x, 0, 0, 0, 0, 1.0f);

  // dt is the delta and we have one delta value per token.
  //           d_inner
  // token 0 [0  ...  7]
  // token 1 [0  ...  7]     seq_len
  // token 2 [0  ...  7]
  // token 3 [0  ...  7]
  struct ggml_tensor* dt = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_inner, seq_len);
  ggml_set_name(dt, "delta");

  // A is the learned state transition matrix.
  //           d_state
  //       0 [0  ...  7]
  //       1 [0  ...  7]     d_inner
  //       2 [0  ...  7]
  //       3 [0  ...  7]
  //       4 [0  ...  7]
  //       5 [0  ...  7]
  //       6 [0  ...  7]
  //       7 [0  ...  7]
  struct ggml_tensor* A = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_state, d_inner);
  ggml_set_name(A, "A");

  // B is the dynamic (not here but in a real Mamba model) input transition matrix.
  //           d_state
  // token 0 [0  ...  7]
  // token 1 [0  ...  7]     seq_len
  // token 2 [0  ...  7]
  // token 3 [0  ...  7]
  struct ggml_tensor* B = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_state, seq_len);
  ggml_set_name(B, "B");

  // C is the output transition matrix.
  //           d_state
  // token 0 [0  ...  7]
  // token 1 [0  ...  7]     seq_len
  // token 2 [0  ...  7]
  // token 3 [0  ...  7]
  struct ggml_tensor* C = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_state, seq_len);
  ggml_set_name(C, "C");

  // r is the result of the scan operation.
  struct ggml_tensor* r = ggml_ssm_scan(ctx, s, x, dt, A, B, C);
  ggml_set_name(r, "result");

  struct ggml_cgraph* c_graph = ggml_new_graph(ctx);
  ggml_build_forward_expand(c_graph, r);

  int n_threads = 1;
  enum ggml_status st = ggml_graph_compute_with_ctx(ctx, c_graph, n_threads);
  if (st != GGML_STATUS_SUCCESS) {
    printf("could not compute graph\n");
    return 1;
  }

  printf("r nelements: %lld:\n", ggml_nelements(r));

  ggml_free(ctx);
  return 0;
}

