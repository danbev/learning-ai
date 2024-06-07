// The intention of this test is to check the availability of the SSSE3
// macro is being set correctly. This is sensitive to the order of options
// specified to the compiler. For example, if -mssse3 is specified and later
// -mavx is specified, the SSSE3 macro will be defined which can be somewhat
// surprising. This is not really an issue with this simple test and one
// Makefile but in a larger project using CMake and including multiple
// directories all with their own CMakeLists.txt files, it can be difficult
// to ensure that the correct flags are being passed to the compiler. Whan even
// more concerning is that you probably won't notice this issue until you
// run the code or inspect it.

#include <stdio.h>

#ifdef __SSSE3__
#error "SSSE3 is defined"
#endif

#ifdef __AVX__
#error "AVX is defined"
#endif

int main() {
  return 0;
}

