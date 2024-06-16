#include <stdio.h>

#include "ggml/ggml.h"
#include "ggml-quants.h"
#include "ggml-impl.h"

#ifdef GGML_COMMON_DECL_C
#warning "[danbev] GGML_COMMON_DECL_C is defined"
#endif

#ifdef GGML_COMMON_DECL_CUDA
#warning "[danbev] GGML_COMMON_DECL_CUDA is defined"
#endif

int main(int argc, char **argv) {
  printf("GGML Quantization examples\n");
  
  float data[] = {0.2, 0.3, 0.4, 0.5};

  float d = 0.5 / 15.0;
  block_q4_0 block_q4_0 = {
        .d = ggml_compute_fp32_to_fp16(d),
        .qs = { 0.2/d, 0.3/d, 0.3/d, 0.4/d, 0.5/d, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
        }
    };
  printf("block_q4_0 delta: %f\n", ggml_compute_fp16_to_fp32(block_q4_0.d));
  for (int i = 0; i < QK4_0/2; i++) {
    printf("block_q4_0.qs[%d]: %d\n", i, block_q4_0.qs[i]);
  }

  printf("Dequantize block_q4_0\n");
  for (int i = 0; i < QK4_0/2; i++) {
    printf("data[%d]: %f\n", i, block_q4_0.qs[i] * ggml_compute_fp16_to_fp32(block_q4_0.d));
  }

  ggml_type_traits_t q4_0 = ggml_internal_get_type_traits(GGML_TYPE_Q4_0);
  printf("ggml type trait name: %s\n", q4_0.type_name);
  printf("ggml type trait block size: %d\n", q4_0.blck_size);
  printf("ggml type trait is_quantized: %s\n", q4_0.is_quantized ? "true" : "false");

  return 0;
}
