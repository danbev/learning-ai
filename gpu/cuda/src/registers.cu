#include <cuda_runtime.h>
#include <stdio.h>

#define N 16

__global__ void registers_only(const int * in, int * out) {
    int tid = threadIdx.x;
    int x = in[tid];   // scalar -> register
    int y = x * 2 + 1; // scalar -> register

    int arr[N];
#pragma unroll
    for (int i = 0; i < N; i++) {
        arr[i] = x + i; // constant index after unrolling -> register
    }

    int sum = 0;
#pragma unroll
    for (int i = 0; i < N; i++) {
        sum += arr[i] * y;
    }

    out[tid] = sum;
}

__global__ void local_dynamic_index(const int * in, const int * idx, int * out) {
    int tid = threadIdx.x;
    int x = in[tid]; // still a register

    int hist[N]; // local memory (look for STL/LDL in the SASS)
#pragma unroll
    for (int i = 0; i < N; i++) {
        hist[i] = 0;
    }

    for (int k = 0; k < 32; k++) {
        int j = idx[(tid + k) % 32] % N; // runtime index
        hist[j] += x;
    }

    int sum = 0;
#pragma unroll
    for (int i = 0; i < N; i++) {
        sum += hist[i] * (i + 1);
    }

    out[tid] = sum;
}

// An array that is simply too big to keep in registers. A thread can use at
// most 255 registers and this needs 256 x 4 bytes = 1KB.
#define BIG 256
__global__ void local_large_array(const int* in, int* out) {
    int tid = threadIdx.x;
    int x = in[tid];

    int big[BIG];
    for (int i = 0; i < BIG; i++) {
        big[i] = x ^ i;
    }

    int sum = 0;
    for (int i = 0; i < BIG; i += (x & 3) + 1) {
        sum += big[i];
    }

    out[tid] = sum;
}

#define CHECK(call)                                                        \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                              \
            return 1;                                                      \
        }                                                                  \
    } while (0)

int main() {
    const int threads = 32;
    int h_in[threads];
    int h_idx[threads];
    int h_out[threads];

    for (int i = 0; i < threads; i++) {
        h_in[i] = i;
        h_idx[i] = (i * 7) % N;
    }

    int * d_in;
    int * d_idx;
    int * d_out;
    CHECK(cudaMalloc(&d_in,  sizeof(h_in)));
    CHECK(cudaMalloc(&d_idx, sizeof(h_idx)));
    CHECK(cudaMalloc(&d_out, sizeof(h_out)));

    CHECK(cudaMemcpy(d_in,   h_in, sizeof(h_in),  cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_idx, h_idx, sizeof(h_idx), cudaMemcpyHostToDevice));

    registers_only<<<1, threads>>>(d_in, d_out);
    CHECK(cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost));
    printf("registers_only:      out[1] = %d\n", h_out[1]);

    local_dynamic_index<<<1, threads>>>(d_in, d_idx, d_out);
    CHECK(cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost));
    printf("local_dynamic_index: out[1] = %d (idx=%d)\n", h_out[1], h_idx[1]);

    local_large_array<<<1, threads>>>(d_in, d_out);
    CHECK(cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost));
    printf("local_large_array:   out[1] = %d\n", h_out[1]);

    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_idx));
    CHECK(cudaFree(d_out));
    return 0;
}
