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

  /*
    GGML_API struct ggml_tensor * ggml_ssm_scan(
            struct ggml_context * ctx,
            struct ggml_tensor  * s,
            struct ggml_tensor  * x,
            struct ggml_tensor  * dt,
            struct ggml_tensor  * A,
            struct ggml_tensor  * B,
            struct ggml_tensor  * C);
   */
  // sequence lenght: 4
  // input embedding dimension: 8 
  // state dimension: 16
  // s is the output of the input->projection->convolution->silu
  int d_state = 16;
  int d_inner = 8;
  int seq_len = 4;
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
  struct ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_inner, seq_len);
  ggml_set_name(s, "x");
  ggml_set_f32_nd(x, 0, 0, 0, 0, 1.0f);

  // dt is the delta and we have one delta value per token.
  struct ggml_tensor* dt = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_inner, seq_len);

  // A is the learned state transition matrix.
  struct ggml_tensor * A = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 16, d_inner);

  // B is the dynamic (not here but in a real Mamba model) input transition matrix.
  struct ggml_tensor * B = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_state, seq_len);

  // C is the output transition matrix.
  struct ggml_tensor * C = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_state, seq_len);

  struct ggml_tensor* r = ggml_ssm_scan(ctx, s, x, dt, A, B, C);

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

