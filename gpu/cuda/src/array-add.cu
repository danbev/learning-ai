#include <stdio.h>

__global__ void add_arrays(int *a, int *b, int *c, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    printf("GPU: adding i = %d\n", i);
    if (i < size) {
        c[i] = a[i] + b[i];
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

    dim3 grid(1);
    dim3 blocks(N);
    add_arrays<<<grid, blocks>>>(d_a, d_b, d_c, N);

    cudaDeviceSynchronize();

    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "array_add kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        return 1;
    }

    // Copy the array that the device has incremented back to the host
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

