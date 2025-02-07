#include <stdio.h>

// Kernal function that runs on the GPU
__global__ void add_arrays(int* a, int* b, int* c, int size) {
    printf("blockIdx.x = %d, blockDim.x = %d, threadIdx.x = %d\n", blockIdx.x, blockDim.x, threadIdx.x);
    // Calculate the index of array index that this thread will process.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] + b[idx];
        printf("[GPU] array index [%d]: adding %d + %d = %d\n", idx, a[idx], b[idx], c[idx]);
    }
}

int main() {
    const int N = 6;
    int size = N * sizeof(int);

    int* h_a = new int[N];
    int* h_b = new int[N];
    int* h_c = new int[N];
    for (int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = i;
    }

    printf("array a:\n");
    for (int i = 0; i < N; i++) {
        printf("%d ", h_a[i]);
    }
    printf("\n");
    printf("array b:\n");
    for (int i = 0; i < N; i++) {
        printf("%d ", h_b[i]);
    }
    printf("\n");

    int* d_a;
    int* d_b;
    int* d_c;
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    dim3 blocks(2);  // blocks per grid
    dim3 threads(3); // threads per block
    add_arrays<<<blocks, threads>>>(d_a, d_b, d_c, N);

    cudaDeviceSynchronize();

    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "array_add kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        return 1;
    }

    // Copy the array that the device has computed back to the host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    printf("Added on GPU:\n");
    for (int i = 0; i < N; i++) {
        printf("%d ", h_c[i]);
    }
    printf("\n");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    return 0;
}

