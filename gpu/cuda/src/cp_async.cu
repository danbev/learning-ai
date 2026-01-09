#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdint>

// This example demonstrates using the cp.async instruction in CUDA to perform
// asynchronous copies from global memory to shared memory. The motivation for
// this is that it avoids an intermediate moveing of data to registers and then
// to shared memory. And it is asynchronous, to the thread can continue doing
// work. This can be useful in a pipelined kernel where computation can be overlapped
// with memory copies.

__device__ __forceinline__ uint32_t cvta_to_shared_u32(const void* ptr) {
    // cvta stands for "convert address"
    // The ptr is 64-bit generic pointer. The GPU can inspect the address bits
    // and determine what type of memory it is pointing to (global, shared, local, etc).
    // But specific hardware units like Async Copy engine don't have this capability
    // , they are highly optimized for specific tasks and just want raw 32-bit
    // offsets into the shared memory (notice that _shared in the instruction, it
    // already knows where this is pointing to and therefor only needs the offset
    // of the generic pointer, not the which memory space, it knows this is for
    // shared memory).
    // Also, a 64-bit pointer requires 2 registers (since CUDA registers are 32-bit).
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    // So this is bacially casting away the high 32-bits, the memory space information
    // of the pointer and is similar to (but safer):
    //return (uint32_t)(uintptr_t)ptr;
}

__device__ __forceinline__ void cp_async_16(void* smem_ptr, const void* global_ptr) {
    // Convert Shared Memory pointer to 32-bit integer offset
    uint32_t smem_int = cvta_to_shared_u32(smem_ptr);
    
    // For the global pointer, we cast to uintptr_t (64-bit int) for the "l" constraint
    uintptr_t global_int = reinterpret_cast<uintptr_t>(global_ptr);

    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(smem_int), "l"(global_int)
        // notice that two :: means that we don't have any outputs!
        // r is a 32 bit register and maps to .u32 or s.32 in PTX.
        // l is a 64 bit long and a global memory address pointer. .u64 or .s64 in PTX.
        // And the immediate value 16 indicates we are copying 16 bytes (4 integers).
        // So this is passing the shared memory location and the global memory
        // location to the cp.async instruction.
        //
    );
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;");
    // Each time we call this we add an entry to a queue which is FIFO.
    // We can then later wait on these entries using the function below.
    // If we wait for all like the current example does it is equivalent to
    // a blocking copy. But we can also wait for 1 which will only take the
    // first entry from the queue (the first commit we issued) and the other
    // is still pending. This allows overlapping of computation and memory
    // copies. So we wait for the oldest but let newest request keep loading.
}

__device__ __forceinline__ void cp_async_wait_all() {
    // This means sleep until 0 groups are left pending.
    asm volatile("cp.async.wait_group 0;");
}

__global__ void async_copy_kernel(const int* global_in, int* global_out, int N) {
    extern __shared__ int tile[];

    int tid = threadIdx.x;
    
    if (tid * 4 < N) {
        int* smem_dest = tile + (tid * 4);
        const int* global_src = global_in + (tid * 4);

        // Issue the async copy operation. This is not queued but is immediately
        // sent to the memory controller. It does not wait for the commit below
        // which is what I initially thought.
        cp_async_16(smem_dest, global_src);
        
        cp_async_commit();
    }

    cp_async_wait_all();
    
    __syncthreads();

    // verify
    if (tid * 4 < N) {
        // Read from shared memory
        int val0 = tile[tid*4 + 0];
        int val1 = tile[tid*4 + 1];
        int val2 = tile[tid*4 + 2];
        int val3 = tile[tid*4 + 3];

        global_out[tid*4 + 0] = val0;
        global_out[tid*4 + 1] = val1;
        global_out[tid*4 + 2] = val2;
        global_out[tid*4 + 3] = val3;
    }
}

int main() {
    printf("Async Copy (cp_async) example.\n");
    
    int num_elements = 128;
    int bytes = num_elements * sizeof(int);

    int* h_in  = (int*)malloc(bytes);
    int* h_out = (int*)malloc(bytes);

    for(int i=0; i<num_elements; i++) {
        h_in[i] = i * 10;
    }

    int *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);

    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    async_copy_kernel<<<1, 32, 512>>>(d_in, d_out, num_elements);

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    printf("Verifying index 0: %d (Expected 0)\n", h_out[0]);
    printf("Verifying index 127: %d (Expected 1270)\n", h_out[127]);

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);
    
    return 0;
}
