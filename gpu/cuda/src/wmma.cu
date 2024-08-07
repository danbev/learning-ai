#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <cstdio>

#define MATRIX_SIZE 16
#define WMMA_TILE_SIZE 16

using namespace nvcuda::wmma;

// This is the CUDA kernel
__global__ void wmma_example_kernel(const half* a, const half* b, half* c) {
    // Define the fragment types for the input and output matrices
    fragment<matrix_a, WMMA_TILE_SIZE, WMMA_TILE_SIZE, WMMA_TILE_SIZE, half, row_major> a_frag;
    fragment<matrix_b, WMMA_TILE_SIZE, WMMA_TILE_SIZE, WMMA_TILE_SIZE, half, col_major> b_frag;
    fragment<accumulator, WMMA_TILE_SIZE, WMMA_TILE_SIZE, WMMA_TILE_SIZE, half> c_frag;
    // Initialize the output to zero
    fill_fragment(c_frag, __float2half(0.0f));

    // Load the input matrices into the fragments
    load_matrix_sync(a_frag, a, MATRIX_SIZE);
    load_matrix_sync(b_frag, b, MATRIX_SIZE);

    // Perform the matrix multiplication
    mma_sync(c_frag, a_frag, b_frag, c_frag);

    // Store the result back to memory
    store_matrix_sync(c, c_frag, MATRIX_SIZE, mem_row_major);
}

int main() {
    printf("Warp-level Matrix Muliply Accumulate example.\n\n");
    // Initialize host matrices
    half h_a[MATRIX_SIZE * MATRIX_SIZE];
    half h_b[MATRIX_SIZE * MATRIX_SIZE];
    half h_c[MATRIX_SIZE * MATRIX_SIZE];

    // Fill matrices with example data
    for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; ++i) {
        h_a[i] = __float2half(1.0f);
        h_b[i] = __float2half(1.0f);
    }

    // Allocate device memory
    half *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, MATRIX_SIZE * MATRIX_SIZE * sizeof(half));
    cudaMalloc((void**)&d_b, MATRIX_SIZE * MATRIX_SIZE * sizeof(half));
    cudaMalloc((void**)&d_c, MATRIX_SIZE * MATRIX_SIZE * sizeof(half));

    // Copy host matrices to device
    cudaMemcpy(d_a, h_a, MATRIX_SIZE * MATRIX_SIZE * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, MATRIX_SIZE * MATRIX_SIZE * sizeof(half), cudaMemcpyHostToDevice);

    // Launch the WMMA kernel
    wmma_example_kernel<<<1, 32>>>(d_a, d_b, d_c);

    // Copy the result back to host
    cudaMemcpy(h_c, d_c, MATRIX_SIZE * MATRIX_SIZE * sizeof(half), cudaMemcpyDeviceToHost);

    // Print the result matrix
    for (int i = 0; i < MATRIX_SIZE; ++i) {
        for (int j = 0; j < MATRIX_SIZE; ++j) {
            printf("%.0f ", __half2float(h_c[i * MATRIX_SIZE + j]));
        }
	printf("\n");
    }

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
