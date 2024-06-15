#include <stdio.h>
#include <stdint.h>

#define DANBEV_QK4_0 32

typedef struct {
    float d;                       // delta
    uint8_t qs[DANBEV_QK4_0 / 2];  // nibbles / quants
} danbev_block_q4_0;

void quantize(float* input, int length, danbev_block_q4_0* output) {
    // Calculate the delta (scaling factor)
    float max_val = 0.0f;
    for (int i = 0; i < length; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }
    output->d = max_val / 15.0;  // 15 is the max value for 4-bit (1111b)

    printf("max_val = %f, d = %f\n", max_val, output->d);

    // Quantize the values
    for (int i = 0; i < length; i++) {
        uint8_t quantized_val = (uint8_t) (input[i] / output->d);
        printf("input[%d] = %f, quantized_val = %d\n", i, input[i], quantized_val);
        if (i % 2 == 0) {
            output->qs[i / 2] = quantized_val << 4;  // Store in the higher nibble
        } else {
            output->qs[i / 2] |= quantized_val;      // Store in the lower nibble
        }
    }
}

void dequantize(danbev_block_q4_0 *input, float *output, int length) {
    for (int i = 0; i < length; i++) {
        uint8_t quantized_val;
        if (i % 2 == 0) {
            quantized_val = input->qs[i / 2] >> 4;  // Extract the higher nibble(4-bits)
        } else {
            quantized_val = input->qs[i / 2] & 0x0F;  // Extract the lower nibble (4-bits)
        }
        output[i] = quantized_val * input->d;
    }
}

void print_nibbles(uint8_t *qs, int length) {
    for (int i = 0; i < length; i++) {
        uint8_t byte = qs[i];
        uint8_t high_nibble = byte >> 4;  // Extract the higher nibble
        uint8_t low_nibble = byte & 0x0F; // Extract the lower nibble

        printf("Byte %d: high nibble = %u, low nibble = %u\n", i, high_nibble, low_nibble);
    }
}

#define Q_SIZE 4

int main(int argc, char **argv) {
  printf("GGML Quantization examples\n");
  
  float data[Q_SIZE] = {0.2, 0.3, 0.4, 0.5};

  danbev_block_q4_0 quantized;
  float dequantized[Q_SIZE];

  quantize(data, Q_SIZE, &quantized);
  print_nibbles(quantized.qs, Q_SIZE/2);

  return 0;
}
