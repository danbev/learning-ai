#include <stdio.h>
#include <immintrin.h>

int main() {
    // Initialize two 256-bit floating-point vectors
    __m256 vec1 = _mm256_setr_ps(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
    __m256 vec2 = _mm256_setr_ps(8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0);

    // The following performs:
    // vec1    [ lane 1] [ lane 2]
    // hadd1 = {1+2, 3+4, 5+6, 7+8}
    // hadd1 = {3, 7,     11, 15}

    // vec2    [ lane 1] [ lane 2]
    // hadd2 = {8+7, 6+5, 4+3, 2+1}
    // hadd2 = {15, 11,    7, 3}

    // This result is interleaved as follows:
    // [vec1 lane 1] [vec2 lane 1] [vec1 lane 2] [vec2 lane 2]
    // {   3   7       15   11       11   15    7    3}
    __m256 hadd1 = _mm256_hadd_ps(vec1, vec2);
    printf("hadd1:\n");
    float* res = (float*)&hadd1;
    for (int i = 0; i < 8; i++) {
        printf("%f ", res[i]);
    }
    printf("\n");
    printf("vec1 hadd:\n");
    res = (float*)&hadd1;
    for (int i = 0; i < 4; i++) {
        printf("%f ", res[i]);
    }
    printf("\n");
    printf("vec2 hadd:\n");
    res = (float*)&hadd1;
    for (int i = 4; i < 8; i++) {
        printf("%f ", res[i]);
    }
    printf("\n");

    // Split into two 128-bit parts and perform another horizontal add
    // 3.0   7.0   15.0   11.0     11.0   15.0   7.0   3.0
    //         [ lane 1]    [ lane 2]
    // hadd1 = {3+7, 15+11  11+15, 7+3}
    // hadd1 = {10   26     26     10}
    //
    // hadd2 = {3+7, 15+11  11+15, 7+3}
    // hadd2 = {10   26     26     10}
    ///
    // This result is interleaved as follows:
    // [vec1 lane 1] [vec2 lane 1] [vec1 lane 2] [vec2 lane 2]
    // {  10  26      10   26        26    10      26   10}
    __m256 hadd2 = _mm256_hadd_ps(hadd1, hadd1);  // Horizontal add of hadd1 parts

    // Display the elements of the resulting vector
    printf("Result added:\n");
    res = (float*)&hadd2;
    for (int i = 0; i < 8; i++) {
        printf("%f ", res[i]);
    }
    printf("\n");

    return 0;
}

