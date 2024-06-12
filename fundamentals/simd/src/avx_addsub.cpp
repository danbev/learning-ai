#include <immintrin.h>
#include <iostream>

int main() {
    // __m256d is a 256 bit type that can hold 4 double precision (64 bit) floats
    // _mm256_setr_pd sets in reverse order.
    __m256d a = _mm256_setr_pd(1.0, 2.0, 3.0, 4.0);
    printf("vector a: [%f, %f, %f, %f]\n", a[0], a[1], a[2], a[3]);

    __m256d b = _mm256_setr_pd(5.0, 6.0, 7.0, 8.0);
    printf("vector b: [%f, %f, %f, %f]\n", b[0], b[1], b[2], b[3]);

    // Perform add/sub operation alternating subtraction and addition:
    // a[0] - b[0]
    // a[1] + b[1]
    // a[2] - b[2]
    // a[3] + b[3]
    // Now I found this to be confusing since the name of this operation is
    // addsub and not subadd. But the reason for this naming or the order of
    // operations is because is stems from complex number operations.
    // TODO: add notes about this.
    __m256d result = _mm256_addsub_pd(a, b);

    double* f = (double*)&result;
    printf("Result: [%f, %f, %f, %f]\n", f[0], f[1], f[2], f[3]);
    return 0;
}
