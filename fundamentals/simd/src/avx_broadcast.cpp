#include <immintrin.h>
#include <iostream>

int main() {
    double vb = 3.0;
    __m256d broadcasted = _mm256_broadcast_sd(&vb);

    double result[4];
    _mm256_storeu_pd(result, broadcasted);
    printf("Result: [%f, %f, %f, %f]\n", result[0], result[1], result[2], result[3]);
    return 0;
}
