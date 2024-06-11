#include <immintrin.h>
#include <iostream>

void vector_add(float *a, float *b, float *c, int n) {
    // Process 8 floats at a time using AVX
    int i;
    for (i = 0; i < n; i += 8) {
        // Load 8 floats from each array into AVX registers
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);

        // Perform the addition
        __m256 vc = _mm256_add_ps(va, vb);

        // Store the result back into the result array
        _mm256_storeu_ps(&c[i], vc);
    }
}

int main() {

    int n = 16;

    // Allocate aligned memory for arrays
    float a[16] __attribute__((aligned(32)));
    float b[16] __attribute__((aligned(32)));
    float c[16] __attribute__((aligned(32)));

    // Initialize arrays with some values
    for (int i = 0; i < n; i++) {
        a[i] = i * 1.0f;
        b[i] = i * 2.0f;
    }

    // Perform vector addition
    vector_add(a, b, c, n);

    // Print the results
    for (int i = 0; i < n; i++) {
        printf("c[%d] = %f\n", i, c[i]);
    }

    return 0;
}
