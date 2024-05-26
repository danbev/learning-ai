#include <stdio.h>
#include <immintrin.h>

int main() {
    int i;
    int int_array[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    
    __m256i mask = _mm256_setr_epi32(-1, -1, -1, -1, 0, 0, 0, 0);
    
    __m256i result = _mm256_maskload_epi32(int_array, mask);
    
    int* res = (int*)&result;
    printf("%d %d %d %d %d %d %d %d\n", 
        res[0], res[1], res[2], res[3], res[4], res[5], res[6], res[7]);
    
    return 0;
}

