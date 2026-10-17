#include <cuda_runtime.h>

#include <cstdio>
#include <algorithm>
#include <cstdint>

__device__ float nvfp4_to_float(uint8_t nibble) {
    static const float lut[16] = {
	 0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
    };
    return lut[nibble & 0x0F];
}

__global__ void kernel(
		const uint8_t * A,
		const uint8_t * B,
		float * C,
		float global_scale) {
    // Each thread handles 2 elements (one byte)
    int idx = threadIdx.x;
    uint8_t byte_a = A[idx];
    uint8_t byte_b = B[idx];

    float a_low = nvfp4_to_float(byte_a & 0x0F);
    float b_low = nvfp4_to_float(byte_b & 0x0F);

    float a_high = nvfp4_to_float(byte_a >> 4);
    float b_high = nvfp4_to_float(byte_b >> 4);

    float dot = (a_low * b_low) + (a_high * b_high);
    printf("thread [idx]: %d dot product %f\n", idx, dot);

    // Apply the Global Scale
    C[idx] = dot * global_scale;
}

int main() {
    printf("Blackwell sm_121 Manual Dequantization Path\n\n");

    const int num_bytes = 8; // 16 elements as each byte holds 2, 4-bit values
    uint8_t h_a[num_bytes];
    uint8_t h_b[num_bytes];
    float   h_c[num_bytes];

    float global_scale = 2.0f;

    for(int i=0; i < num_bytes; ++i) {
	// (cuda-gdb) x /1bt &h_a[0]
        // 0xffffffffee98: 0110 0110
	//                 SEEM SEEM S=0, E=11, M=0 which gives 2^(3-1) * 1.0 = 4.0
        h_a[i] = 0x66; // Represents [4.0, 4.0]

	// (cuda-gdb) p /t h_b[0]
        // $30 = 0010 0010
	// (cuda-gdb) x /1bt &h_b[0]
        // 0xffffffffeea0: 0010 0010
	//                 SEEM SEEM S=0, E=01, M=0 which gives 2^(2-1) * 1.0 = 1.0
        h_b[i] = 0x22; // Represents [1.0, 1.0]
    }
    // So we will have:
    // (cuda-gdb) p /t h_a
    // $31 = {1100110, 1100110, 1100110, 1100110, 1100110, 1100110, 1100110, 1100110} 
    // (cuda-gdb) p /t h_b
    // $32 = {0100010, 0100010, 0100010, 0100010, 0100010, 0100010, 0100010, 0100010}

    uint8_t * d_a;
    uint8_t * d_b;
    float   * d_c;
    cudaMalloc(&d_a, num_bytes);
    cudaMalloc(&d_b, num_bytes);
    cudaMalloc(&d_c, num_bytes * sizeof(float));

    cudaMemcpy(d_a, h_a, num_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, num_bytes, cudaMemcpyHostToDevice);

    kernel<<<1, num_bytes>>>(d_a, d_b, d_c, global_scale);

    cudaMemcpy(h_c, d_c, num_bytes * sizeof(float), cudaMemcpyDeviceToHost);

    // Result should be: (4.0 * 1.0 + 4.0 * 1.0) * 2.0 = 16.0
    printf("Result of first byte: %.1f (Expected 16.0)\n", h_c[0]);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return 0;
}
