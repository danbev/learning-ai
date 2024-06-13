#include <immintrin.h>
#include <cstdio>

int main() {
    __m256d vec1 = _mm256_set1_pd(1.0);
    __m256d vec2 = _mm256_set1_pd(0.0);
    __m256d result = _mm256_and_pd(vec1, vec2);
    printf("Result: [%f]\n", result[0]);
    // The result now contains the bitwise AND of the binary representations of 10.5 and 20.25
    return 0;
}

