#include <immintrin.h>
#include <cstdio>

int main() {
    printf("_mm256_permutevar8x32_epi32 examples:\n\n");

    // Create an array of 32-bit integers
    int input[8] = {10, 20, 30, 40, 50, 60, 70, 80};
    __m256i vec = _mm256_loadu_si256((__m256i*)input);
    
    // Create different permutation patterns (indices)
    // Each element can be 0-7, specifying which input element to take
    
    // Example 1: Reverse the order
    int reverse_indices[8] = {7, 6, 5, 4, 3, 2, 1, 0};
    __m256i reverse_perm = _mm256_loadu_si256((__m256i*)reverse_indices);
    __m256i result1 = _mm256_permutevar8x32_epi32(vec, reverse_perm);
    
    // Example 2: Duplicate some elements
    int duplicate_indices[8] = {0, 0, 1, 1, 2, 2, 3, 3};
    __m256i duplicate_perm = _mm256_loadu_si256((__m256i*)duplicate_indices);
    __m256i result2 = _mm256_permutevar8x32_epi32(vec, duplicate_perm);
    
    // Example 3: Custom rearrangement (like the ggml requiredOrder pattern)
    int custom_indices[8] = {3, 2, 1, 0, 7, 6, 5, 4};
    __m256i custom_perm = _mm256_loadu_si256((__m256i*)custom_indices);
    __m256i result3 = _mm256_permutevar8x32_epi32(vec, custom_perm);
    
    // Extract results
    int output1[8], output2[8], output3[8];
    _mm256_storeu_si256((__m256i*)output1, result1);
    _mm256_storeu_si256((__m256i*)output2, result2);
    _mm256_storeu_si256((__m256i*)output3, result3);
    
    printf("Input vector:\n");
    printf("Position: ");
    for (int i = 0; i < 8; i++) printf("%d   ", i);
    printf("\nValue:    ");
    for (int i = 0; i < 8; i++) printf("%d  ", input[i]);
    
    printf("\n\nExample 1 - Reverse order:\n");
    printf("Indices:  ");
    for (int i = 0; i < 8; i++) printf("%d   ", reverse_indices[i]);
    printf("\nResult:   ");
    for (int i = 0; i < 8; i++) printf("%d  ", output1[i]);
    printf("\n(Takes input[7], input[6], input[5], etc.)\n");
    
    printf("\nExample 2 - Duplicate elements:\n");
    printf("Indices:  ");
    for (int i = 0; i < 8; i++) printf("%d   ", duplicate_indices[i]);
    printf("\nResult:   ");
    for (int i = 0; i < 8; i++) printf("%d  ", output2[i]);
    printf("\n(Takes input[0] twice, input[1] twice, etc.)\n");
    
    printf("\nExample 3 - Custom rearrangement:\n");
    printf("Indices:  ");
    for (int i = 0; i < 8; i++) printf("%d   ", custom_indices[i]);
    printf("\nResult:   ");
    for (int i = 0; i < 8; i++) printf("%d  ", output3[i]);
    printf("\n(Swaps the two halves and reverses each half)\n");
    
    printf("\nThe pattern: result[i] = input[indices[i]]\n");
    printf("Each index can be 0-7, allowing any rearrangement!\n");
    
    return 0;
}
