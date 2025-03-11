#include <stdio.h>

__global__ void matrixMul(float* A, float* B, float* C, int width) {
    // Calculate the row and column index of the element
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread will calculate on element of the output matrix C.
    // So A0 * B0 = C(0,0)
    // So A0 * B1 = C(0,1)
    printf("row: %d, col: %d\n", row, col);
    
    if (row < width && col < width) {
        float sum = 0.0f;
        
        // Compute the dot product of row of A and column of B
        for (int k = 0; k < width; k++) {
            // A[row, k] * B[k, col]
            // A and B are in global memory and will need to be read.
            // And A0 will be used by multiple threads as it is used to calculate
            // C(0,0), C(0,1), C(0,2), C(0,3). And likewise for the other rows.
            // This means that there will be more global memory reads than necessary.
            // This is where shared memory can help. 
            sum += A[row * width + k] * B[k * width + col];
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
    matrixMul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, width);
    
    // Wait for GPU to finish
    cudaDeviceSynchronize();
    
    // Check for errors
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "matrixMul kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
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
