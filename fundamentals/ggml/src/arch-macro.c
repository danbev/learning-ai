#include <stdio.h>

// Architecture check
#if defined(__x86_64__)
    #define ARCH_INFO "x86_64"
#elif defined(__i386__)
    #define ARCH_INFO "x86"
#elif defined(__arm__)
    #define ARCH_INFO "ARM"
#elif defined(__aarch64__)
    #define ARCH_INFO "ARM64"
#else
    #define ARCH_INFO "Unknown"
#endif

// SIMD feature checks
#if defined(__ARM_NEON)
    #define ARM_NEON_INFO "__ARM_NEON: Yes"
#else
    #define ARM_NEON_INFO "__ARM_NEON: No"
#endif

#if defined(__ARM_FEATURE_FMA)
    #define ARM_FMA_INFO "__ARM_FEATURE_FMA: Yes"
#else
    #define ARM_FMA_INFO "__ARM_FEATURE_FMA: No"
#endif

#if defined(__SSE__)
    #define SSE_INFO "__SSE__: Yes"
#else
    #define SSE_INFO "__SSE__: No"
#endif

#if defined(__SSE2__)
    #define SSE2_INFO "__SSE2__: Yes"
#else
    #define SSE2_INFO "__SSE2__: No"
#endif

#if defined(__AVX__)
    #define AVX_INFO "__AVX__: Yes"
#else
    #define AVX_INFO "__AVX__: No"
#endif

#if defined(__AVX2__)
    #define AVX2_INFO "__AVX2__: Yes"
#else
    #define AVX2_INFO "__AVX2__: No"
#endif

int main() {
    printf("--- Architecture Information ---\n\n");
    printf("Architecture: %s\n\n", ARCH_INFO);
    
    printf("SIMD Features:\n");
    printf(" - %s\n", ARM_NEON_INFO);
    printf(" - %s\n", ARM_FMA_INFO);
    printf(" - %s\n", SSE_INFO);
    printf(" - %s\n", SSE2_INFO);
    printf(" - %s\n", AVX_INFO);
    printf(" - %s\n", AVX2_INFO);
    
    printf("----------------------------\n");
    return 0;
}
