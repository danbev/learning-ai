#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <iostream>

union PackedInt8 {
    int32_t as_int;        // The 32-bit view (what __dp4a sees)
    struct {
        int8_t x, y, z, w;
    } bytes;
};

__device__ __forceinline__ int compute_dp4a(int packed_a, int packed_b, int accumulator) {
    // __dp4a(op1, op2, op3)
    // op1: 32-bit int containing 4 signed 8-bit ints
    // op2: 32-bit int containing 4 signed 8-bit ints
    // op3: 32-bit int accumulator
    // Returns: (op1.byte[0] * op2.byte[0] + 
    //           op1.byte[1] * op2.byte[1] +
    //           op1.byte[2] * op2.byte[2] +
    //           op1.byte[3] * op2.byte[3]) +
    //           op3
    return __dp4a(packed_a, packed_b, accumulator);
}

// Calculate the dot product of two vectors A and B.
// Instead of one thread processing one number, one thread processes 4 pairs at once.
__global__ void dp4a_example_kernel(const int* vec_a, const int* vec_b, int* vec_out, int N_packed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < N_packed) {
        int a_local = vec_a[idx];
        int b_local = vec_b[idx];

        int result = compute_dp4a(a_local, b_local, 0);
        // I'm using zero as the initial accumulator, but this can be useful
        // when we have larger vectors and want to calculate the dot product
        // for all of them. We can do that in a loop:
        // int acc = 0;
        // for (int i = 0; i < num_packets; ++i) {
        //   acc = compute_dp4a(vec_a[i], vec_b[i], acc);
        // }
        // Simliar to a fused multiply-add operation.

        vec_out[idx] = result;
    }
}

void check_gpu_power_status() {
    int deviceCount = 0;

    // If my eGPU is off, this will return an error code.
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess) {
        printf("Reminder: Is the GPU is switched on?\n");
        printf("Reason: %s\n", cudaGetErrorString(error_id));

        exit(EXIT_FAILURE);
    }

    // Double check: The API might succeed but report 0 devices
    if (deviceCount == 0) {
        printf("No CUDA-capable devices detected.\n");
        printf("Reminder: Is the GPU is switched on?\n");
        exit(EXIT_FAILURE);
    }

    printf("GPU Status: Online (%d device%s found)\n", deviceCount, deviceCount > 1 ? "s" : "");
}

int main() {
    printf("Exploring __dp4a intrinsic...\n");

    check_gpu_power_status();

    const int num_packets = 1; 

    PackedInt8 h_a, h_b;

    // Vector A: [1, 2, 3, 4]
    h_a.bytes.x = 1;
    h_a.bytes.y = 2;
    h_a.bytes.z = 3;
    h_a.bytes.w = 4;

    // Vector B: [10, 20, 30, 40]
    h_b.bytes.x = 10;
    h_b.bytes.y = 20;
    h_b.bytes.z = 30;
    h_b.bytes.w = 40;

    printf("Input A (8-bit): [%d, %d, %d, %d] (Packed as int: %d)\n", h_a.bytes.x, h_a.bytes.y, h_a.bytes.z, h_a.bytes.w, h_a.as_int);
    printf("Input B (8-bit): [%d, %d, %d, %d] (Packed as int: %d)\n", h_b.bytes.x, h_b.bytes.y, h_b.bytes.z, h_b.bytes.w, h_b.as_int);

    int * d_a;
    cudaMalloc(&d_a, sizeof(int) * num_packets);

    int * d_b;
    cudaMalloc(&d_b, sizeof(int) * num_packets);

    int * d_out;
    cudaMalloc(&d_out, sizeof(int) * num_packets);

    cudaMemcpy(d_a, &h_a.as_int, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &h_b.as_int, sizeof(int), cudaMemcpyHostToDevice);

    dp4a_example_kernel<<<1, 1>>>(d_a, d_b, d_out, num_packets);

    int h_out;
    cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);

    int expected = (1*10) + (2*20) + (3*30) + (4*40);

    printf("GPU Result: %d\n", h_out);
    printf("CPU Result: %d\n", expected);

    if (h_out == expected) {
        printf("Success: __dp4a calculated correctly.\n");
    } else {
        printf("Failure: Calculation mismatch.\n");
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    return 0;
}
