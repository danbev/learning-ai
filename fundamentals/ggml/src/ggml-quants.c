#include <stdio.h>

#include "ggml.h"
#include "ggml-quants.h"
#include "ggml-impl.h"

#ifdef GGML_COMMON_DECL_C
#warning "[danbev] GGML_COMMON_DECL_C is defined"
#endif

#ifdef GGML_COMMON_DECL_CUDA
#warning "[danbev] GGML_COMMON_DECL_CUDA is defined"
#endif

#define Q_SIZE 4

int main(int argc, char **argv) {
  printf("GGML Quantization examples\n");
  
  float data[Q_SIZE] = {0.2, 0.3, 0.4, 0.5};

  float d = 0.5 / 15.0;
  block_q4_0 block_q4_0 = {
        .d = ggml_compute_fp32_to_fp16(d),
        .qs = { 0.2/d, 0.3/d, 0.3/d, 0.4/d, 0.5/d, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
        }
    };
  printf("block_q4_0 delta: %f\n", ggml_compute_fp16_to_fp32(block_q4_0.d));
  printf("block_q4_0 qs:\n");
  for (int i = 0; i < Q_SIZE; i++) {
    printf("block_q4_0.qs[%d]: %d\n", i, block_q4_0.qs[i]);
  }

  return 0;
}
