#include <cuda_runtime.h>

#include <cstdio>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
            return 1; \
        } \
    } while (0)


__global__ void trigger_misaligned_address(char * d_ptr) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid == 0) {
        // Create a pointer offset by 8 bytes.
        // cudaMalloc guarantees 256-byte alignment, so (d_ptr + 8) ends in 0x...8.
        // 8-byte aligment means addresses must start a 0, 8, 16, 24, ...
        char * misaligned_addr = d_ptr + 8;
        printf("d_ptr address: %p\n", d_ptr);
        printf("Misaligned address (d_ptr + 8): %p\n", misaligned_addr);
        printf("\n");

        printf("Kernel: Attempting to read 16 bytes (int4) from address: %p\n", misaligned_addr);
        
        // Cast to int4* (which requires 16-byte alignment).
        // 16-byte alignment means addresses must start a 0, 16, 32, 48, ...
        // This is asking the GPU to load a single 16-byte chunk (int4) starting
        // at offset 8. But the hardware expects to be able to grab perfectly aligned
        // 16-byte chunks (0-15, 16-31, etc). But starting at 8 would mean it has
        // to grab bytes 8-23, which crosses the 16-byte boundary. But it refuses
        // to do this, and instead raises a 'Warp Misaligned Address' exception.
        // misaligned address: 0x7e65aac00008
        // So the following would be asking for a 16 bit chunk starting:
        // 8-23 but this the chunks the instruction and hardware want are
        // 0-15, 16-31, etc.
        int4 * vec_ptr = (int4*) misaligned_addr;

        // But we are just dealing with memory here and we could instead do
        // something like this:
        //
        int2 * vec_ptr_fixed = (int2*) misaligned_addr;

        int2 val1 = vec_ptr_fixed[0]; // This is fine, reads bytes 8-15
        int2 val2 = vec_ptr_fixed[1]; // This is fine, reads bytes 16-23

        // Dereference.
        // This generates a 'LD.E.128' instruction (Load 128-bit / 16-byte).
        // Since the address is 8-byte aligned, not 16-byte aligned, this causes
        // CUDA Exception: Warp Misaligned Address.
        int4 value = *vec_ptr; 
        
        // workaround using two int2 reads instead
        int4 v;
        v.x = val1.x;
        v.y = val1.y;
        v.z = val2.x;
        v.w = val2.y;

        // If we reach this point then we did not generated the expected exception.
        printf("Kernel: Read successful: %d %d %d %d\n", value.x, value.y, value.z, value.w);
    }
}

int main() {
    printf("Misaligned address reproduction...\n");

    // Allocate device memory (guaranteed to be aligned to at least 256 bytes)
    char * d_data;
    CUDA_CHECK(cudaMalloc((void**)&d_data, 1024));

    // Initialize memory so we aren't reading garbage (if it were to succeed)
    CUDA_CHECK(cudaMemset(d_data, 0, 1024));

    trigger_misaligned_address<<<1, 1>>>(d_data);

    cudaError_t err = cudaDeviceSynchronize();

    if (err == cudaSuccess) {
        printf("Success: Kernel finished without error (Unexpected).\n");
    } else {
        printf("\nFAILURE DETECTED:\n");
        printf("-----------------\n");
        printf("Caught Error: %s\n", cudaGetErrorString(err));
        printf("This confirms the 'Warp Misaligned Address' exception.\n");
    }

    cudaFree(d_data);
    return 0;
}
