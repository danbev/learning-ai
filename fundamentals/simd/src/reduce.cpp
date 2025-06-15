#include <immintrin.h>
#include <stdio.h>

int main() {
    constexpr int SIZE = 8;
    __m256 x[SIZE];
    
    x[0] = _mm256_setr_ps( 1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f);
    x[1] = _mm256_setr_ps( 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f);
    x[2] = _mm256_setr_ps(17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f);
    x[3] = _mm256_setr_ps(25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f, 32.0f);
    x[4] = _mm256_setr_ps(33.0f, 34.0f, 35.0f, 36.0f, 37.0f, 38.0f, 39.0f, 40.0f);
    x[5] = _mm256_setr_ps(41.0f, 42.0f, 43.0f, 44.0f, 45.0f, 46.0f, 47.0f, 48.0f);
    x[6] = _mm256_setr_ps(49.0f, 50.0f, 51.0f, 52.0f, 53.0f, 54.0f, 55.0f, 56.0f);
    x[7] = _mm256_setr_ps(57.0f, 58.0f, 59.0f, 60.0f, 61.0f, 62.0f, 63.0f, 64.0f);
    
    printf("Before reduction:\n");
    for (int i = 0; i < SIZE; i++) {
        float vals[8];
        _mm256_storeu_ps(vals, x[i]);
        printf("x[%d]: ", i);
        for (int j = 0; j < 8; j++) {
            printf("%5.1f ", vals[j]);
        }
        printf("\n");
    }
    
    int offset = SIZE >> 1; // Start with half the size (4 pairs)
    for (int i = 0; i < offset; ++i) {
        // Add pairs of vectors. So this is summing the first 4 vectors with the last 4 vectors.
        // 
        // (gdb) p x[0]
        // $12 = {1, 2, 3, 4, 5, 6, 7, 8}
        // (gdb) p x[offset+0]
        // $13 = {33, 34, 35, 36, 37, 38, 39, 40}
        // (gdb) p x[0]
        // $14 = {34, 36, 38, 40, 42, 44, 46, 48}
        x[i] = _mm256_add_ps(x[i], x[offset+i]);
    }

    offset >>= 1;           // Reduce size by half again (2 pairs)
    for (int i = 0; i < offset; ++i) {
        // Add pairs of vectors. So this is summing the first 2 vectors with the last 2 vectors.
        x[i] = _mm256_add_ps(x[i], x[offset+i]);
    }

    offset >>= 1;           // Reduce size by half again (1 pair)
    for (int i = 0; i < offset; ++i) {
        // Add pairs of vectors. So this is summing the first vector with the last vector.
        x[i] = _mm256_add_ps(x[i], x[offset+i]);
    }

    // Extract/cast the lower 128 bits from the 256-bit vector
    // (gdb) p x[0]
    // $1 = {232, 240, 248, 256, 264, 272, 280, 288}
    const __m128 lower = _mm256_castps256_ps128(x[0]);
    // (gdb) p lower
    // $2 = {232, 240, 248, 256}
    // Extract 128-bit from the 256-bit vector, and specify that we want the upper part (1)
    const __m128 upper = _mm256_extractf128_ps(x[0], 1);
    // (gdb) p upper
    // $4 = {264, 272, 280, 288}
    const __m128 t0 = _mm_add_ps(lower, upper);
    // (gdb) p t0
    // $5 = {496, 512, 528, 544}

    // Horizontal add of the elements in t0, and are only interested in the first
    // pair but we have to pass something to __mm_hadd_ps as it expects two vectors
    // and it is common to just specify the same vector twice, but we could have
    // used an empty (zeroed) vector as well.
    __m128 t1 = _mm_hadd_ps(t0, t0);
    // (gdb) p t1
    // $6 = {1008, 1072, 1008, 1072}
    t1 = _mm_hadd_ps(t1, t1);

    // Extract/convert the lowest single precision float value from a 128-bit vector.
    float result = (float) _mm_cvtss_f32(t1);
    
    printf("\nAfter reduction:\n");
    printf("Final result: %.1f\n", result);
    
    // Manual verification: sum of all 64 numbers (1+2+...+64 = 64*65/2 = 2080)
    printf("Expected result: %.1f\n", 64.0f * 65.0f / 2.0f);
    
    return 0;
}
