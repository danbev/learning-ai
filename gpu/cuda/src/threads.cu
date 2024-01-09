#include <npp.h>
#include <stdio.h>

// CUDA Kernel function to print "Hello World"
__global__ void threads() {
    int grid_dim = gridDim.x;
    int block_dim = blockDim.x;
    printf("GPU: grid_dim: %d, block_dim: %d\n", grid_dim, block_dim);
    int block_id = blockIdx.x;
    int thread_id = threadIdx.x;
    printf("GPU: block_id: %d, thread_id: %d\n", block_id, thread_id);
}

int main() {
    dim3 grid(2, 1, 1);
    dim3 blocks(2, 1, 1);
    threads<<<grid, blocks>>>();

    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "threads launch failed: %s\n", cudaGetErrorString(cudaStatus));
        return 1;
    }


    // Wait for GPU to finish before accessing on host
    // which is for printf to sync the output buffer with the host
    // or else the order of the output will be messed up.
    cudaDeviceSynchronize();

    printf("CPU: threads example\n");

    return 0;
}

