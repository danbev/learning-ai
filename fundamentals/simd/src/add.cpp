#include <immintrin.h>
#include <iostream>

int main() {
    int array1[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    int array2[8] = {8, 7, 6, 5, 4, 3, 2, 1};
    int result[8] = {0};

    __m256i a = _mm256_loadu_si256((__m256i*)array1);
    __m256i b = _mm256_loadu_si256((__m256i*)array2);

    __m256i c = _mm256_add_epi32(a, b);  // _epi32 denotes 32-bit integer vectors

    _mm256_storeu_si256((__m256i*)result, c);

    for (int i = 0; i < 8; i++) {
        std::cout << "result[" << i << "] = " << result[i] << std::endl;
    }

    return 0;
}

