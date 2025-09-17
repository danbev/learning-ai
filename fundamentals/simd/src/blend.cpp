#include <immintrin.h>
#include <cstdio>

int main() {
    printf("Simple _mm256_blend_epi32 example\n\n");

    // Create two arrays of 32-bit integers
    int array_a[8] = {100, 101, 102, 103, 104, 105, 106, 107};
    int array_b[8] = {200, 201, 202, 203, 204, 205, 206, 207};
    
    __m256i vec_a = _mm256_loadu_si256((__m256i*)array_a);
    __m256i vec_b = _mm256_loadu_si256((__m256i*)array_b);
    
    // The blend mask: each bit controls one 32-bit element
    // Bit = 0: take from first vector (vec_a)
    // Bit = 1: take from second vector (vec_b)
    
    // Example 1: mask = 170 (binary: 10101010)
    // This means: take from B, A, B, A, B, A, B, A
    __m256i result1 = _mm256_blend_epi32(vec_a, vec_b, 170);
    
    // Example 2: mask = 240 (binary: 11110000)  
    // This means: take from B, B, B, B, A, A, A, A
    __m256i result2 = _mm256_blend_epi32(vec_a, vec_b, 240);
    
    // Example 3: mask = 85 (binary: 01010101)
    // This means: take from A, B, A, B, A, B, A, B  
    __m256i result3 = _mm256_blend_epi32(vec_a, vec_b, 85);
    
    // Extract results
    int output1[8], output2[8], output3[8];
    _mm256_storeu_si256((__m256i*)output1, result1);
    _mm256_storeu_si256((__m256i*)output2, result2);
    _mm256_storeu_si256((__m256i*)output3, result3);
    
    
    printf("Input vectors:\n");
    printf("A: ");
    for (int i = 0; i < 8; i++) printf("%d ", array_a[i]);
    printf("\nB: ");
    for (int i = 0; i < 8; i++) printf("%d ", array_b[i]);
    
    printf("\n\nExample 1 - mask = 170 (10101010 binary):\n");
    for (int i = 0; i < 8; i++) printf("%d ", output1[i]);
    
    printf("\n\nExample 2 - mask = 240 (11110000 binary):\n");
    for (int i = 0; i < 8; i++) printf("%d ", output2[i]);
    
    printf("\n\nExample 3 - mask = 85 (01010101 binary):\n");
    for (int i = 0; i < 8; i++) printf("%d ", output3[i]);
    
    printf("\n\nThe pattern: bit i controls element i\n");
    printf("0 = take from first vector, 1 = take from second vector\n");
    
    return 0;
}
