## Single Instruction, Multiple Data (SIMD)

### Introduction
SIMD utilizes special registers which can hold 128, 256, 512, or even 1024 bits
of data. The register used is divided into smaller blocks of 8, 16, 32, or 64
bits and perform the same operation on all the blocks simultaneously.

### Machine specific options/support
If you just try to compile a program using SIMD instructions, you may get the
following error message:
```console
$ make add
gcc -g -Wall  -Wno-unused-variable   -o bin/add src/add.cpp
src/add.cpp: In function ‘int main()’:
src/add.cpp:11:52: warning: AVX vector return without AVX enabled changes the ABI [-Wpsabi]
   11 |     __m256i a = _mm256_loadu_si256((__m256i*)array1);
      |                                                    ^
In file included from /usr/lib/gcc/x86_64-linux-gnu/11/include/immintrin.h:43,
                 from src/add.cpp:1:
/usr/lib/gcc/x86_64-linux-gnu/11/include/avxintrin.h:933:1: error: inlining failed in call to ‘always_inline’ ‘void _mm256_storeu_si256(__m256i_u*, __m256i)’: target specific option mismatch
  933 | _mm256_storeu_si256 (__m256i_u *__P, __m256i __A)
      | ^~~~~~~~~~~~~~~~~~~
src/add.cpp:18:24: note: called from here
   18 |     _mm256_storeu_si256((__m256i*)result, c);
      |     ~~~~~~~~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~
In file included from /usr/lib/gcc/x86_64-linux-gnu/11/include/immintrin.h:47,
                 from src/add.cpp:1:
/usr/lib/gcc/x86_64-linux-gnu/11/include/avx2intrin.h:119:1: error: inlining failed in call to ‘always_inline’ ‘__m256i _mm256_add_epi32(__m256i, __m256i)’: target specific option mismatch
  119 | _mm256_add_epi32 (__m256i __A, __m256i __B)
      | ^~~~~~~~~~~~~~~~
src/add.cpp:15:33: note: called from here
   15 |     __m256i c = _mm256_add_epi32(a, b);  // _epi32 denotes 32-bit integer vectors
      |                 ~~~~~~~~~~~~~~~~^~~~~~
In file included from /usr/lib/gcc/x86_64-linux-gnu/11/include/immintrin.h:43,
                 from src/add.cpp:1:
/usr/lib/gcc/x86_64-linux-gnu/11/include/avxintrin.h:927:1: error: inlining failed in call to ‘always_inline’ ‘__m256i _mm256_loadu_si256(const __m256i_u*)’: target specific option mismatch
  927 | _mm256_loadu_si256 (__m256i_u const *__P)
      | ^~~~~~~~~~~~~~~~~~
src/add.cpp:12:35: note: called from here
   12 |     __m256i b = _mm256_loadu_si256((__m256i*)array2);
      |                 ~~~~~~~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~
In file included from /usr/lib/gcc/x86_64-linux-gnu/11/include/immintrin.h:43,
                 from src/add.cpp:1:
/usr/lib/gcc/x86_64-linux-gnu/11/include/avxintrin.h:927:1: error: inlining failed in call to ‘always_inline’ ‘__m256i _mm256_loadu_si256(const __m256i_u*)’: target specific option mismatch
  927 | _mm256_loadu_si256 (__m256i_u const *__P)
      | ^~~~~~~~~~~~~~~~~~
src/add.cpp:11:35: note: called from here
   11 |     __m256i a = _mm256_loadu_si256((__m256i*)array1);
      |                 ~~~~~~~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~
make: *** [Makefile:13: add] Error 1
```
In this case we nee to add the 'm'achine specific option `avx2` flag to the
compilation command:
```console
$ g++ -g -Wall -mavx2 -Wno-unused-variable -o bin/add src/add.cpp
```
