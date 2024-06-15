#include <immintrin.h>
#include <iostream>

bool isAligned32(void* ptr) {
    return (uintptr_t) ptr % 32 == 0;
}

int main() {
    double vb = 3.0;
    printf("is vb aliged: %s\n", isAligned32(&vb) ? "true" : "false");

    __m256d broadcasted = _mm256_broadcast_sd(&vb);

    double result[4];
    // store unaligned
    _mm256_storeu_pd(result, broadcasted);
    printf("Unaligned Result: [%f, %f, %f, %f]\n", result[0], result[1], result[2], result[3]);

    alignas(32) double vb2 = 3.0;
    printf("Is vb2 aliged: %s\n", isAligned32(&vb2) ? "true" : "false");
    __m256d broadcasted2 = _mm256_broadcast_sd(&vb2);
    alignas(32) double result2[4];
    // store aligned
    _mm256_store_pd(result2, broadcasted);
    printf("Aligned Result: [%f, %f, %f, %f]\n", result2[0], result2[1], result2[2], result2[3]);

    return 0;
}
