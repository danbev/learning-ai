#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdio>
#include <cstdint>

// This example demonstrates how to use the ldmatrix instruction which I came
// accross in ggml-cuda which is used in the CUDA flash attention implementation.

// This mimics types like T_B_KQ and is a 16x16 tile of FP16 elements = 256 elements.
// Distributed across 32 threads in a warp. Each thread holds:
// 256 elements / 32 threads   = 8 elements.
// 8 elements * 2 bytes (fp16) = 16 bytes per thread.
// 16 bytes                    = 4x uint32_t registers.
struct Tile_16x16 {
    uint32_t x[4];
};

// This is a helper function similar to what exist in ggml and uses CUDA inline
// assembly to issue the ldmatrix instruction.
// The format is as follows:
// asm("template-string" : "constraint"(output) : "constraint"(input));
//
static __device__ __forceinline__ void load_ldmatrix(Tile_16x16 &dst, const half *ptr, int stride_in_bytes) {
    // Get the shared memory address as a simple integer
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));

    // 2. The ldmatrix instruction
    // We are loading 4 registers (.x4 part of the instruction) from shared memory.
    // This fills the 'dst.x' array with data formatted for Tensor Cores.
    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(dst.x[0]), "=r"(dst.x[1]), "=r"(dst.x[2]), "=r"(dst.x[3])
        : "r"(smem_addr)
    );
}

__global__ void ldmatrix_kernel(const half* global_in, float* global_out) {
    
    extern __shared__ half tile_smem[];

    //Load from Global Memory -> Shared Memory
    const int tid = threadIdx.x;
    
    // Each thread loads 8 elements (using int4 = 16 bytes copy)
    // We cast to int4 to copy faster
    if (tid < 32) {
        const int4* src = reinterpret_cast<const int4*>(global_in);
        int4* dst = reinterpret_cast<int4*>(tile_smem);
        dst[tid] = src[tid];
    }

    // Barrier: Wait for shared memory (tile_smem) to be populated.
    __syncthreads();

    // Create our register tile
    Tile_16x16 Q_B;

    load_ldmatrix(Q_B, tile_smem + (tid * 8), 0);

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> frag_a;
    
    // Copy our raw registers into the standard fragment
    uint32_t* frag_ptr = reinterpret_cast<uint32_t*>(&frag_a);
    frag_ptr[0] = Q_B.x[0];
    frag_ptr[1] = Q_B.x[1];
    frag_ptr[2] = Q_B.x[2];
    frag_ptr[3] = Q_B.x[3];

    // Create an Identity matrix for B so A * B = A
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> frag_b;
    nvcuda::wmma::fill_fragment(frag_b, 0.0f); // Reset
    nvcuda::wmma::fill_fragment(frag_b, 1.0f);

    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> frag_c;
    nvcuda::wmma::fill_fragment(frag_c, 0.0f);

    // Compute
    nvcuda::wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);

    // Store Output
    nvcuda::wmma::store_matrix_sync(global_out, frag_c, 16, nvcuda::wmma::mem_row_major);
}

int main() {
    printf("ldmatrix example...\n");

    const int N = 16 * 16;
    const int bytes_in = N * sizeof(half);
    const int bytes_out = N * sizeof(float);

    half* h_in = (half*)malloc(bytes_in);
    float* h_out = (float*)malloc(bytes_out);
    
    // Fill input with 1.0
    for(int i=0; i<N; i++) h_in[i] = __float2half(1.0f);

    half* d_in;
    float* d_out;
    cudaMalloc(&d_in, bytes_in);
    cudaMalloc(&d_out, bytes_out);

    cudaMemcpy(d_in, h_in, bytes_in, cudaMemcpyHostToDevice);

    // Dynamic Shared Memory: 16x16x2 bytes = 512 bytes
    ldmatrix_kernel<<<1, 32, 512>>>(d_in, d_out);

    cudaMemcpy(h_out, d_out, bytes_out, cudaMemcpyDeviceToHost);

    printf("Top left value: %.2f (Expected 16.00)\n", h_out[0]);

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);
    return 0;
}
