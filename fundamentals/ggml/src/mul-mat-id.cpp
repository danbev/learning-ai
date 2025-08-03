#include <stdio.h>

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

int main(int argc, char **argv) {
  printf("GGML mul-mat-id example\n");

  struct ggml_init_params params = {
    .mem_size   = 16*1024*1024,
    .mem_buffer = NULL,
  };
  struct ggml_context* ctx = ggml_init(params);

  int n_experts = 3;
  int n_feat = 2;
  int rows = 2;
  int n_tokens = 1;
  int n_expert_used = 2;

  // We have 3 experts and each expert has a 2x2 matrix.
  // 'as` as this is an extension of the ggml_mul_mat operation but with
  // multiple experts but keeping the "same" parameter names.
  struct ggml_tensor* as = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_feat, rows, n_experts);
  ggml_set_name(as, "a");
  printf("as->ne[0]: %ld\n", as->ne[0]);
  printf("as->ne[1]: %ld\n", as->ne[1]);
  printf("as->ne[2]: %ld\n", as->ne[2]);

  // Expert matrix 0
  ggml_set_f32_nd(as, 0, 0, 0, 0, 1);
  ggml_set_f32_nd(as, 1, 0, 0, 0, 0);
  ggml_set_f32_nd(as, 0, 1, 0, 0, 0);
  ggml_set_f32_nd(as, 1, 1, 0, 0, 1);
  // Expert matrix 1
  ggml_set_f32_nd(as, 0, 0, 1, 0, 2);
  ggml_set_f32_nd(as, 1, 0, 1, 0, 0);
  ggml_set_f32_nd(as, 0, 1, 1, 0, 0);
  ggml_set_f32_nd(as, 1, 1, 1, 0, 2);
  // Expert matrix 2
  ggml_set_f32_nd(as, 0, 0, 2, 0, 1);
  ggml_set_f32_nd(as, 1, 0, 2, 0, 2);
  ggml_set_f32_nd(as, 0, 1, 2, 0, 3);
  ggml_set_f32_nd(as, 1, 1, 2, 0, 4);
  for (int e = 0; e < as->ne[2]; e++) {
      printf("expert %d:\n", e);
      for (int y = 0; y < as->ne[1]; y++) {
          for (int x = 0; x < as->ne[0]; x++) {
              printf("%.2f ", *(float *) ((char *) as->data + e * as->nb[2] + y * as->nb[1] + x * as->nb[0]));
           }
          printf("\n");
      }
  }
  printf("\n");

  // This input data tensor, and has the same number of colums as the expert
  // matrices, and y specifies that we are dealing with two experts.
  struct ggml_tensor* b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_feat, n_expert_used, n_tokens);
  ggml_set_name(b, "b");
  ggml_set_f32_nd(b, 0, 0, 0, 0, 0.1);
  ggml_set_f32_nd(b, 1, 0, 0, 0, 0.2);
  ggml_set_f32_nd(b, 0, 1, 0, 0, 0.3);
  ggml_set_f32_nd(b, 1, 1, 0, 0, 0.4);

  for (int e = 0; e < b->ne[2]; e++) {
      printf("token %d:\n", e);
      for (int y = 0; y < b->ne[1]; y++) {
          for (int x = 0; x < b->ne[0]; x++) {
              printf("%.2f ", *(float *) ((char *) b->data + e * b->nb[2] + y * b->nb[1] + x * b->nb[0]));
           }
          printf("\n");
      }
  }
  printf("\n");

  // This tensor specifies which expert to use for each token.
  struct ggml_tensor* ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, n_tokens);
  ggml_set_name(ids, "ids");
  // First token (which expert it should use):
  ggml_set_i32_nd(ids, 0, 0, 0, 0, 0);   // first token
  ggml_set_i32_nd(ids, 1, 0, 0, 0, 2);   // expert to use

  //ggml_set_i32_nd(ids, 0, 1, 0, 0, 2);   // second token
  //ggml_set_i32_nd(ids, 1, 1, 0, 0, 0);   // expert to use
  printf("ids:\n");
  for (int y = 0; y < ids->ne[1]; y++) {
      for (int x = 0; x < ids->ne[0]; x++) {
          printf("%d ", *(int *) ((char *) ids->data + y * ids->nb[1] + x * ids->nb[0]));
       }
      printf("\n");
  }

  struct ggml_tensor* result = ggml_mul_mat_id(ctx, as, b, ids);
  ggml_set_name(result, "result");

  struct ggml_cgraph* c_graph = ggml_new_graph(ctx);
  ggml_build_forward_expand(c_graph, result);
  int n_threads = 4;
  enum ggml_status st = ggml_graph_compute_with_ctx(ctx, c_graph, n_threads);
  if (st != GGML_STATUS_SUCCESS) {
    printf("could not compute graph\n");
    return 1;
  }

  printf("result tensor type: %s\n", ggml_type_name(result->type));
  printf("result dim: %d\n", ggml_n_dims(result));
  printf("result dim[0]: %ld\n", result->ne[0]);
  printf("result dim[1]: %ld\n", result->ne[1]);
  printf("result dim[2]: %ld\n", result->ne[2]);
  printf("result data:\n");
  for (int e = 0; e < result->ne[2]; e++) {
      for (int y = 0; y < result->ne[1]; y++) {
          for (int x = 0; x < result->ne[0]; x++) {
              printf("%.2f ", *(float *) ((char *) result->data + e * result->nb[2] + y * result->nb[1] + x * result->nb[0]));
           }
          printf("\n");
      }
  }
  printf("\n");

  ggml_free(ctx);
  return 0;
}
