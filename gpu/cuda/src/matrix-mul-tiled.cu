#include <stdio.h>

#define TILE_SIZE 4

__global__ void matrixMulTiled(float* A, float* B, float* C, int width) {
    __shared__ float sharedA[TILE_SIZE][TILE_SIZE];
    __shared__ float sharedB[TILE_SIZE][TILE_SIZE];
    
    // Calculate the row and column index of the element
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize accumulator for this thread's output element
    float sum = 0.0f;
    
    // Loop through tiles along the k-dimension
    // For 8x8 matrix with 4x4 tiles: need 8/4 = 2 tile phases
    for (int tileIdx = 0; tileIdx < width / TILE_SIZE; tileIdx++) {

        // PHASE 1: Cooperative loading of current tiles
        // Load tile from matrix A
        if (row < width && (tileIdx * TILE_SIZE + threadIdx.x) < width) {
            sharedA[threadIdx.y][threadIdx.x] = A[row * width + (tileIdx * TILE_SIZE + threadIdx.x)];
        } else {
            sharedA[threadIdx.y][threadIdx.x] = 0.0f;  // Padding for boundary conditions
        }

        // Load tile from matrix B
        if (col < width && (tileIdx * TILE_SIZE + threadIdx.y) < width) {
            sharedB[threadIdx.y][threadIdx.x] = B[(tileIdx * TILE_SIZE + threadIdx.y) * width + col];
        } else {
            sharedB[threadIdx.y][threadIdx.x] = 0.0f;  // Padding for boundary conditions
        }

        // PHASE 2: Synchronize to make sure the tiles are loaded
        __syncthreads();

        // PHASE 3: Compute partial dot product using shared memory
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += sharedA[threadIdx.y][k] * sharedB[k][threadIdx.x];
        }

        // PHASE 4: Synchronize before loading next tile
        __syncthreads();
    }
    
    // Write final result to global memory
    if (row < width && col < width) {
        C[row * width + col] = sum;
    }
}

int main() {
    int width = 8;
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
    printf("Matrix A (8x8):\n");
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            printf("%.1f ", h_A[i * width + j]);
        }
        printf("\n");
    }
    
    printf("\nMatrix B (8x8):\n");
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
    
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);  // 4x4 = 16 threads per block
    dim3 blocksPerGrid((width + TILE_SIZE - 1) / TILE_SIZE,
                       (width + TILE_SIZE - 1) / TILE_SIZE);  // 2x2 = 4 blocks total

    printf("\nGrid configuration:\n");
    printf("Threads per block: (%d, %d) = %d threads\n",
           threadsPerBlock.x, threadsPerBlock.y,
           threadsPerBlock.x * threadsPerBlock.y);
    printf("Blocks per grid: (%d, %d) = %d blocks\n",
           blocksPerGrid.x, blocksPerGrid.y,
           blocksPerGrid.x * blocksPerGrid.y);
    printf("Total threads: %d\n",
           threadsPerBlock.x * threadsPerBlock.y * blocksPerGrid.x * blocksPerGrid.y);

    // Launch the kernel
    matrixMulTiled<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, width);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Check for errors
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "matrixMulTiled kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        return 1;
    }

    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Print the result
    printf("\nMatrix C (8x8 result):\n");
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
