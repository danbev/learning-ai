#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <cstdio>

__global__ void wmma_example_kernel(const half* a, const half* b, half* c) {
    // Define the fragment types for the input and output matrices. These are
    // distributed data structures that are spread accross the threads in a warp.
    // A single thread can have 255 registers per thread and if we have type of
    // like f32 which are 4 bytes that means 255 * 4 = 1020 bytes per thread.
    // And in this case we are going to multiply two 16x16 matrices of f16
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> c_frag;

    // Initialize the output matrix to zero
    nvcuda::wmma::fill_fragment(c_frag, __float2half(0.0f));

    // The following will load the input matrices from gloabl memory into
    // the threads registers. Note that a single thread cannot keep all the
    // elements in its registers, so these are spread out evenly accross the
    // threads in the warp, so 256/32 = 8 elements per thread.
    nvcuda::wmma::load_matrix_sync(a_frag, a, 16);
    nvcuda::wmma::load_matrix_sync(b_frag, b, 16);

    // Perform the matrix multiplication and accumulate the results
    // C += AxB
    nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    // Store the result from the threads registers back to global memory.
    nvcuda::wmma::store_matrix_sync(c, c_frag, 16, nvcuda::wmma::mem_row_major);
}

int main() {
    printf("Warp-level Matrix Muliply Accumulate (WMMA) example.\n\n");

    half h_a[16 * 16];
    half h_b[16 * 16];
    half h_c[16 * 16];

    // Fill matrices with example data
    for (int i = 0; i < 16 * 16; ++i) {
        h_a[i] = __float2half(1.0f);
        h_b[i] = __float2half(2.0f);
    }

    // Allocate device memory
    half *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, 16 * 16 * sizeof(half));
    cudaMalloc((void**)&d_b, 16 * 16 * sizeof(half));
    cudaMalloc((void**)&d_c, 16 * 16 * sizeof(half));

    // Copy host matrices to device
    cudaMemcpy(d_a, h_a, 16 * 16 * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, 16 * 16 * sizeof(half), cudaMemcpyHostToDevice);

    // Launch the WMMA kernel
    wmma_example_kernel<<<1, 32>>>(d_a, d_b, d_c);

    // Copy the result back to host
    cudaMemcpy(h_c, d_c, 16 * 16 * sizeof(half), cudaMemcpyDeviceToHost);

    // Print the result matrix
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 16; ++j) {
            printf("%.0f ", __half2float(h_c[i * 16 + j]));
        }
        printf("\n");
    }

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
