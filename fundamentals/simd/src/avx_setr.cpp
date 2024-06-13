#include <immintrin.h>
#include <iostream>

int main() {
    // set_pd will add 1.0 to the highest element and 4.0 to the lowest element.
    // which is the oposite of what I expected.
    __m256d a = _mm256_set_pd(1.0, 2.0, 3.0, 4.0);
    printf("vector a: [%f, %f, %f, %f]\n", a[0], a[1], a[2], a[3]);

    // setr_pd sets in reverse order where 1.0 will be in the lowest slot
    // and 4.0 will be in the highest slot.
    __m256d b = _mm256_setr_pd(1.0, 2.0, 3.0, 4.0);
    printf("vector b: [%f, %f, %f, %f]\n", b[0], b[1], b[2], b[3]);
    return 0;
}
