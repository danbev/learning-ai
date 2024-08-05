#include <stdio.h>
#include <math.h>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

int main(int argc, char **argv) {
  printf("GGML RoPE example\n");

  struct ggml_init_params params = {
    .mem_size   = 20000000,
    .mem_buffer = NULL,
  };
  struct ggml_context* ctx = ggml_init(params);

  // Simulate a sequence of 6 tokens with en embedding size of 4096 and a
  // context length of 512. 
  int n_ctx_orig = 4096;
  int embd_dim = 128;
  int n_head = 32;
  int n_tokens = 6;

  // The Query matrix in this case can hold 512 tokens each with a dimension
  // of 4096.
  struct ggml_tensor* query = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_ctx_orig, n_tokens);

  // We reshape the query matrix embedding dimensions to account for the number
  // of heads (32) each which will have a dimension of 128 (128 * 32 = 4096).
  struct ggml_tensor* a = ggml_reshape_3d(ctx, query, embd_dim, n_head, n_tokens);
  ggml_set_name(a, "a");

  // These are the positions 
  struct ggml_tensor* pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
  ggml_set_name(pos, "pos");

  // Set some made up values for the tensor to be rotated.
  // First loop over the number of tokens in the batch (6) (skipping the actual
  // loop for the batch here though.
  for (int i = 0; i < a->ne[2]; i++) {
      // Loop over the embedding heads (32)
      for (int j = 0; j < a->ne[1]; j++) {
          // Loop over the embedding dimensions (128)
          for (int k = 0; k < a->ne[0]; k++) {
              float value = 0.0f + k;
              ggml_set_f32_nd(a, k, j, i, 0, value);
          }
      } 
  }

  // Print a few of the first dimensions so we can see that there is a rotation
  // being performed. In this case we are printing the first 10 embeddings for
  // the 2nd token. I'm not using token 0 as this will have a cosine value of 10
  // and since value of 0 which will not perform any rotations for the position
  // embeddings for that dimension.
  for (int i = 0; i < 10; i++) {
    printf("embedding for token 1, embedding dim %d: %f\n", i, ggml_get_f32_nd(a, i, 0, 1, 0));
  }

  // Set the positions manually (the b tensor parameter to ggml_rope_ext).
  for (int i = 0; i < pos->ne[0]; i++) {
      ggml_set_i32_1d(pos, i, i);
  }

  int mode = 0;    // rote type 0 = Normal

  // The RoPE base frequency
  //   ↓
  // (10000^(-2j/d).
  float freq_base = 10000.0f;

  // The RoPE frequency scale.
  float freq_scale = 1.0f;

  // TODO: What is this? It looks like this is mscale (magnituce scale)
  float attn_factor = 1.0f;

  // Extrapolation factor. If this is 0.0 then the beta_fast and beta_slow
  // are not used. 
  float ext_factor = 1.0f;

  // This is a YaRN parameter is named α (alpha) in the YaRN paper. This
  // specifies that hen the number of rotations is 32 this is the position
  // embedding dimension that should be used for the for 
  float beta_fast = 32.0f;

  // This is a YaRN parameter which I think is named β in the YaRN paper.
  float beta_slow = 1.0f;

  // LongRope Frequency factors (freq_factors/rope_scaling) are used with
  // certain models like Phi-3-mini-128k-instruct
  // (https://huggingface.co/microsoft/Phi-3-mini-128k-instruct/blob/main/config.json#L27).
  struct ggml_tensor* freq_factors = NULL;

  struct ggml_tensor* s = ggml_rope_ext(ctx,
                                        a,
                                        pos,
                                        freq_factors,
                                        embd_dim,
                                        mode,
                                        n_ctx_orig,
                                        freq_base,
                                        freq_scale,
                                        ext_factor,
                                        attn_factor,
                                        beta_fast,
                                        beta_slow);

  struct ggml_cgraph* c_graph = ggml_new_graph(ctx);
  ggml_build_forward_expand(c_graph, s);

  int n_threads = 4;
  enum ggml_status status = ggml_graph_compute_with_ctx(ctx, c_graph, n_threads);
  if (status != GGML_STATUS_SUCCESS) {
    printf("Error: %s\n", ggml_status_to_string(status));
    return 1;
  }
  
  for (int i = 0; i < 10; i++) {
    printf("embedding for token 1, embedding dim %d = %f\n", i, ggml_get_f32_nd(s, i, 0, 1, 0));
  }

  ggml_free(ctx);
  return 0;
}
