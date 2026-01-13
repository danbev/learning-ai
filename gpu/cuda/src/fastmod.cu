#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>

static __device__ __forceinline__ uint32_t fastdiv(uint32_t n, const uint3 fastdiv_values) {
    // fastdiv_values.x = Multiplier (m)
    // fastdiv_values.y = Shift (l)
    
    // 1. Calculate high 32-bits of (n * m)
    const uint32_t hi = __umulhi(n, fastdiv_values.x);
    
    // 2. The formula from Hacker's Delight for when the multiplier > 2^32
    return (hi + n) >> fastdiv_values.y;
}

static __device__ __forceinline__ uint32_t fastmodulo(uint32_t n, const uint3 fastdiv_values) {
    // n % d = n - (n / d) * d
    // fastdiv_values.z holds the original divisor 'd'
    return n - fastdiv(n, fastdiv_values) * fastdiv_values.z;
}

__global__ void test_modulo_kernel(const uint32_t* inputs, uint32_t* outputs_ref, uint32_t* outputs_fast, int N, uint3 fastdiv_vals) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        uint32_t val = inputs[tid];

        // 1. Slow Standard Modulo (uses hardware division)
        outputs_ref[tid] = val % fastdiv_vals.z;

        // 2. Fast Magic Modulo (uses multiply + shift)
        outputs_fast[tid] = fastmodulo(val, fastdiv_vals);
    }
}

// This function calculates the "Magic Numbers" to replace division with multiplication.
// Based on "Hacker's Delight", Chapter 10, Unsigned Division by Constants.
uint3 find_magic_numbers(uint32_t d) {
    uint3 result;
    result.z = d; // Store the original divisor

    // 1. Handle edge cases (should not happen in attention, but good for safety)
    if (d == 0) return {0,0,0};
    if (d == 1) return {0,0,1};

    // 2. Algorithm to find Multiplier (m) and Shift (p)
    // We want to find m and p such that: n / d approx (n * m) >> p
    uint32_t p = 31;
    uint32_t m = 0;
    
    // We iterate to find a power of 2 (2^p) such that we can approximate 1/d
    // The specific logic here targets the "(hi + n) >> shift" instruction sequence.
    uint64_t two_power_32 = 1ULL << 32;
    
    // Calculate ceil(log2(d))
    uint32_t l = 0;
    while ((1ULL << l) < d) l++;
    
    // The specific formula for the instruction sequence used in the kernel:
    // This is for divisors where the multiplier exceeds 2^32.
    // m = (2^(32+l) / d) - 2^32 + 1
    // shift = l
    
    uint64_t target = 1ULL << (32 + l);
    // ceil(target / d)
    uint64_t magic_full = (target + d - 1) / d; 
    
    // Because we use the (hi + n) trick, we save the "overflow" part
    result.x = (uint32_t)(magic_full - two_power_32); // The multiplier
    result.y = l;                                     // The shift
    
    return result;
}

int main() {
    const int N = 100;
    // Use a tricky non-power-of-2 divisor (e.g., Sequence Length 57)
    // Power-of-2 divisors are optimized by the compiler automatically.
    const uint32_t divisor = 57; 
    
    printf("Testing Fast Modulo for Divisor: %u\n", divisor);

    // Calculate Magic Numbers on CPU
    uint3 magic = find_magic_numbers(divisor);
    printf("Magic Numbers -> Multiplier(x): %u, Shift(y): %u, Divisor(z): %u\n", 
           magic.x, magic.y, magic.z);

    // 2. Allocate Memory
    std::vector<uint32_t> h_input(N);
    std::vector<uint32_t> h_ref(N);
    std::vector<uint32_t> h_fast(N);
    
    // Fill inputs with random data
    for(int i=0; i<N; i++) {
        h_input[i] = rand();
    }

    uint32_t *d_input, *d_ref, *d_fast;
    cudaMalloc(&d_input, N * sizeof(uint32_t));
    cudaMalloc(&d_ref, N * sizeof(uint32_t));
    cudaMalloc(&d_fast, N * sizeof(uint32_t));

    cudaMemcpy(d_input, h_input.data(), N * sizeof(uint32_t), cudaMemcpyHostToDevice);

    test_modulo_kernel<<<1, 128>>>(d_input, d_ref, d_fast, N, magic);
    cudaDeviceSynchronize();

    // Verify
    cudaMemcpy(h_ref.data(), d_ref, N * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fast.data(), d_fast, N * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    int errors = 0;
    for(int i=0; i<N; i++) {
        if (h_ref[i] != h_fast[i]) {
            printf("Mismatch at %d: Input %u, Ref %% %u = %u, Fast %% %u = %u\n", 
                   i, h_input[i], divisor, h_ref[i], divisor, h_fast[i]);
            errors++;
            if(errors > 5) break;
        }
    }

    if(errors == 0) {
        printf("SUCCESS! All %d values matched.\n", N);
        printf("Example Calculation for Input %u:\n", h_input[0]);
        printf("  Standard: %u %% %u = %u\n", h_input[0], divisor, h_ref[0]);
        printf("  Fast:     %u - ((%u * %u)>>32 + %u) >> %u) * %u = %u\n", 
               h_input[0], h_input[0], magic.x, h_input[0], magic.y, divisor, h_fast[0]);
    } else {
        printf("FAILED with %d errors.\n", errors);
    }

    cudaFree(d_input); cudaFree(d_ref); cudaFree(d_fast);
    return 0;
}
