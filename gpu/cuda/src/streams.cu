#include <cuda_runtime.h>
#include <stdio.h>

#define N 20
#define STREAMS 4

__global__ void vector_add(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;
    cudaStream_t streams[STREAMS];

    // Allocate host memory
    h_a = (float*)malloc(N * sizeof(float));
    h_b = (float*)malloc(N * sizeof(float));
    h_c = (float*)malloc(N * sizeof(float));

    // Allocate device memory
    cudaMalloc((void**)&d_a, N * sizeof(float));
    cudaMalloc((void**)&d_b, N * sizeof(float));
    cudaMalloc((void**)&d_c, N * sizeof(float));

    // Initialize host arrays
    for (int i = 0; i < N; i++) {
        h_a[i] = i+1;
        h_b[i] = i+1;
    }
    printf("A array:\n");
    for (int i = 0; i < N; i++) {
        printf("%.2f\n", h_a[i]);
    }
    printf("B array:\n");
    for (int i = 0; i < N; i++) {
        printf("%.2f\n", h_b[i]);
    }

    // Create streams
    for (int i = 0; i < STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // Divide work among streams
    int shared_mem = 0;
    int threads_per_block = 256;
    int streamSize = N / STREAMS;
    int n_blocks = (streamSize + threads_per_block - 1) / threads_per_block;
    for (int i = 0; i < STREAMS; i++) {
        int offset = i * streamSize;
        cudaMemcpyAsync(&d_a[offset], &h_a[offset], streamSize * sizeof(float), cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync(&d_b[offset], &h_b[offset], streamSize * sizeof(float), cudaMemcpyHostToDevice, streams[i]);
        
        vector_add<<<(n_blocks, threads_per_block, shared_mem, streams[i]>>>(&d_a[offset], &d_b[offset], &d_c[offset], streamSize);
        
        cudaMemcpyAsync(&h_c[offset], &d_c[offset], streamSize * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
    }

    // Wait for all streams to complete
    cudaDeviceSynchronize();

    printf("C array:\n");
    for (int i = 0; i < N; i++) {
        printf("%.2f\n", h_c[i]);
    }

    // Clean up
    for (int i = 0; i < STREAMS; i++) {
        cudaStreamDestroy(streams[i]);
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
