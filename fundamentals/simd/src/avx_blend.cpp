#include <immintrin.h>
#include <cstdio>

int main() {
    __m256d a = _mm256_setr_pd(1.0, 2.0, 3.0, 4.0);
    __m256d b = _mm256_setr_pd(5.0, 6.0, 7.0, 8.0);

    // For the mask 1010 the result will be:
    // result[0] = a[0] because 1 is set in the first bit of the mask
    // result[1] = b[1] because 0 is set in the second bit of the mask
    // result[2] = a[2] because 1 is set in the third bit of the mask
    // result[3] = b[3] because 0 is set in the fourth bit of the mask
    const int mask = 0b1010;
    __m256d result = _mm256_blend_pd(a, b, mask);
    printf("Result: [%f, %f, %f, %f]\n", result[0], result[1], result[2], result[3]);
    return 0;
}

