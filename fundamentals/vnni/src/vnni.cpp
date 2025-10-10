#include <immintrin.h>
#include <iostream>
#include <array>
#include <cstdint>
#include <iomanip>

int main() {
    __m512i vec_a = _mm512_set1_epi8(1);         // Broadcast 1 to all elements of a 512-bit integer vector
    __m512i vec_b = _mm512_set1_epi8(2);         // Broadcast 2 to all elements of another 512-bit integer vector
    __m512i vec_result = _mm512_setzero_epi32(); // Initialize result vector of 32-bit integers to zero
    
    uint8_t a_bytes[64];  // 512 bits = 64 bytes
    uint8_t b_bytes[64];

    // Notice that this is storing all 64 bytes of the 512-bit vector in 
    // one operation.
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(a_bytes), vec_a);
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(b_bytes), vec_b);

    std::cout << "vec_a (as bytes):" << std::endl;
    for (int i = 0; i < 64; i++) {
        std::cout << std::setw(3) << (int)a_bytes[i];
        if ((i + 1) % 16 == 0) std::cout << std::endl;
    }

    std::cout << "\nvec_b (as bytes):" << std::endl;
    for (int i = 0; i < 64; i++) {
        std::cout << std::setw(3) << (int)b_bytes[i];
        if ((i + 1) % 16 == 0) std::cout << std::endl;
    }

    // Perform the dot product and accumulate in 32-bit integers
    vec_result = _mm512_dpbusd_epi32(vec_result, vec_a, vec_b);
    // dp = dot product
    // b  = byte (8 bits)
    // u  = unsigned
    // s  = signed
    // d  = doubleword (32 bits)
    // epi32 = data type of the destination vector (32-bit integers)
    //
    // This will use the instruction VPDPBUSD:
    // V  = vector (indicates that this operates on a vector not a scalar)
    // P  = packed (operates on multiple data points in parallel)
    // DP = dot product (performs a dot product operation)
    // B  = byte (indicates that the first operand is in byte format)
    // U  = unsigned (indicates that the first operand is treated as unsigned)
    // S  = signed (indicates that the second operand is treated as signed)
    // D  = doubleword (indicates that the result is stored in doubleword format)

    int result_array[16];
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(result_array), vec_result);

    std::cout << "\nResult: " << std::endl;
    for (int i = 0; i < 16; i++) {
        std::cout << "[" << i << "]: " << result_array[i] << std::endl;
    }
    std::cout << std::endl;

    return 0;
}
