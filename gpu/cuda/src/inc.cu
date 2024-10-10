#include <stdio.h>

// This example shows how an array can be incremented on a GPU by using
// a kernel function. 

// Notice how this function only updates a single elemenet of the array and this
// is done by a single thread. So we will have to launch a kernel with N threads
// to update the entire array.
__global__ void increment_gpu(int* d_array) {
    int i = threadIdx.x;
    printf("GPU: d_array[%d] = %d + 1\n", i, d_array[i]);
    d_array[i] =  d_array[i] + 1;
}

int main() {
    const int N = 4;	
    // Host array (h))
    int h_array[N] = {0, 1, 2, 3};
    // Device array (d)
    int* d_array;

    cudaMalloc((void**)&d_array, N * sizeof(int));

    // Copy the host array to the device cudaMemCpy(dst, src, size, direction)
    cudaMemcpy(d_array, h_array, N * sizeof(int), cudaMemcpyHostToDevice);

    dim3 grid(1);
    dim3 blocks(N);
    increment_gpu<<<grid, blocks>>>(d_array);

    cudaDeviceSynchronize();

    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "inc kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        return 1;
    }

    // Copy the array that the device has incremented back to the host
    cudaMemcpy(h_array, d_array, N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Incremented on GPU:\n");
    for (int i = 0; i < N; i++) {
        printf("%d ", h_array[i]);
    }
    printf("\n");

    cudaFree(d_array);
    return 0;
}

