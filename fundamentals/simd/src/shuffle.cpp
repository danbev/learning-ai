#include <immintrin.h>
#include <cstdio>
#include <cstdint>

int main() {
    printf("Simple _mm256_shuffle_epi8 example\n");

    // Create a simple lookup table - just letters A through P
    uint8_t lookup_table[32] = {
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P'
    };

    __m256i table = _mm256_loadu_si256( (__m256i*) lookup_table);

    // Create indices - these are what we use to "look up" values
    uint8_t indices[32] = {
        0 ,  1,  2,  3,  4,  5,  6,  7,
        8,   9, 10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23,
        24, 25, 26, 27, 28, 29, 30, 31
    };
    __m256i idx = _mm256_loadu_si256((__m256i*)indices);

    // The shuffle operation will map all values in one single operation
    __m256i result = _mm256_shuffle_epi8(table, idx);

    uint8_t output[32];
    _mm256_storeu_si256((__m256i*)output, result);

    printf("Index -> Letter\n");
    
    for (int i = 0; i < 32; i++) {
        printf("%.2d --> %c\n", (int)indices[i], (char)output[i]);
    }

    return 0;
}
