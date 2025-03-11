#include <stdio.h>

__global__ void matrixMulShared(float* A, float* B, float* C, int width) {
    // The following creates a 2D grid of float values in shared memory which
    // all threads in the block can access. We have one for the values for
    // matrix A and one for the values for matrix B.
    __shared__ float sharedA[4][4];
    __shared__ float sharedB[4][4];
    
    // Calculate the row and column index of the element
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread reads one value from matrix A into the shared memory. Remember
    // that this is all done in parallel so each thread will only read one value
    // from global memory into shared memory.
    // And shared memory is physically located on the SM, so it is much faster.
    if (row < width && threadIdx.x < width) {
        sharedA[threadIdx.y][threadIdx.x] = A[row * width + threadIdx.x];
    }
    
    // Likewise we do the exact same thing for matrix B.
    if (col < width && threadIdx.y < width) {
        sharedB[threadIdx.y][threadIdx.x] = B[threadIdx.y * width + col];
    }
    
    // Synchronize to make sure the matrices are loaded
    __syncthreads();
    
    // Each thread computes one element of the result matrix but now can use
    // the shared memory values instead of global memory values.
    if (row < width && col < width) {
        float sum = 0.0f;
        
        // Compute the dot product using shared memory
        for (int k = 0; k < width; k++) {
            sum += sharedA[threadIdx.y][k] * sharedB[k][threadIdx.x];
        }
        
        C[row * width + col] = sum;
    }
}

int main() {
    int width = 4;
    int size = width * width * sizeof(float);
    
    // Allocate host memory
    float* h_A = new float[width * width];
    float* h_B = new float[width * width];
    float* h_C = new float[width * width];
    
    // Initialize matrices with sample data
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            h_A[i * width + j] = i + j;  // Simple pattern for matrix A
            h_B[i * width + j] = i - j;  // Simple pattern for matrix B
        }
    }
    
    // Print input matrices
    printf("Matrix A:\n");
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            printf("%.1f ", h_A[i * width + j]);
        }
        printf("\n");
    }
    
    printf("\nMatrix B:\n");
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            printf("%.1f ", h_B[i * width + j]);
        }
        printf("\n");
    }
    
    // Allocate device memory
    float* d_A;
    float* d_B;
    float* d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);
    
    // Copy matrices from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    // Define grid and block dimensions
    // For small matrices, we can use a single block
    dim3 threadsPerBlock(width, width);
    dim3 blocksPerGrid(1, 1);
    
    // For larger matrices, we'd use multiple blocks:
    // dim3 threadsPerBlock(16, 16);
    // dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x, 
    //                     (width + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    // Launch the kernel
    matrixMulShared<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, width);
    
    // Wait for GPU to finish
    cudaDeviceSynchronize();
    
    // Check for errors
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "matrixMulShared kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        return 1;
    }
    
    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    // Print the result
    printf("\nMatrix C (result):\n");
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            printf("%.1f ", h_C[i * width + j]);
        }
        printf("\n");
    }
    
    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    // Free host memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    
    return 0;
}
