#include <immintrin.h>
#include <iostream>
#include <array>

bool check_avx512_vnni_support() {
    // store the output registers EAX, EBX, ECX, and EDX
    std::array<int, 4> cpuid_data;

    // EAX = 7, ECX = 0 => Extended Features, a = EAX, b = EBX, c = ECX, d = EDX (output)
    __asm__("cpuid"
            : "=a"(cpuid_data[0]), "=b"(cpuid_data[1]), "=c"(cpuid_data[2]), "=d"(cpuid_data[3])
            : "a"(7), "c"(0));

    // Check EBX[bit 11] for AVX-512 VNNI (also known as AVX-512 VPOPCNTDQ/VNNI)
    return (cpuid_data[1] & (1 << 11)) != 0;
}

/*
   This example willl compile but will not run on my current machine as the
   processor does not support the AVX-512 VNNI instruction set. 
   $ lscpu | grep avx512_vnni

   The output of the program will be:
   Illegal instruction (core dumped)

   Just good to know that this can be the case and see the error.
*/
int main() {
    if (check_avx512_vnni_support()) {
        __m512i vec_a = _mm512_set1_epi8(10);  // Broadcast 10 to all elements of a 512-bit integer vector
        __m512i vec_b = _mm512_set1_epi8(20);  // Broadcast 20 to all elements of another 512-bit integer vector
        __m512i vec_result = _mm512_setzero_epi32();  // Initialize result vector of 32-bit integers to zero

        // Perform the dot product and accumulate in 32-bit integers
        vec_result = _mm512_dpbusd_epi32(vec_result, vec_a, vec_b);

        int result_array[16];  // Create an array to store the results
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(result_array), vec_result);  // Store the vector in the array

        std::cout << "Result of VNNI operation: " << result_array[0] << std::endl;  // Access the first element
    } else {
        std::cout << "AVX-512 VNNI is not supported on this processor." << std::endl;
    }

    return 0;
}
