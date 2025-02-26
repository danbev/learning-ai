#include <stdio.h>

int main() {
    printf("=== ARM Architecture Features ===\n\n");

    // Basic architecture
    #ifdef __ARM_ARCH
        printf("__ARM_ARCH: %d\n", __ARM_ARCH);
    #else
        printf("__ARM_ARCH: Not defined\n");
    #endif

    // Check for various ARM features
    #ifdef __ARM_FEATURE_DSP
        printf("__ARM_FEATURE_DSP: Defined\n");
    #else
        printf("__ARM_FEATURE_DSP: Not defined\n");
    #endif

    #ifdef __ARM_FEATURE_DOTPROD
        printf("__ARM_FEATURE_DOTPROD: Defined\n");
    #else
        printf("__ARM_FEATURE_DOTPROD: Not defined\n");
    #endif

    #ifdef __ARM_FEATURE_CMSE
        printf("__ARM_FEATURE_CMSE: Defined\n");
    #else
        printf("__ARM_FEATURE_CMSE: Not defined\n");
    #endif

    #ifdef __ARM_FEATURE_UNALIGNED
        printf("__ARM_FEATURE_UNALIGNED: Defined\n");
    #else
        printf("__ARM_FEATURE_UNALIGNED: Not defined\n");
    #endif

    #ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        printf("__ARM_FEATURE_FP16_VECTOR_ARITHMETIC: Defined\n");
    #else
        printf("__ARM_FEATURE_FP16_VECTOR_ARITHMETIC: Not defined\n");
    #endif

    #ifdef __ARM_FEATURE_FP16_SCALAR_ARITHMETIC
        printf("__ARM_FEATURE_FP16_SCALAR_ARITHMETIC: Defined\n");
    #else
        printf("__ARM_FEATURE_FP16_SCALAR_ARITHMETIC: Not defined\n");
    #endif

    #ifdef __ARM_FEATURE_COMPLEX
        printf("__ARM_FEATURE_COMPLEX: Defined\n");
    #else
        printf("__ARM_FEATURE_COMPLEX: Not defined\n");
    #endif

    #ifdef __ARM_FEATURE_CRC32
        printf("__ARM_FEATURE_CRC32: Defined\n");
    #else
        printf("__ARM_FEATURE_CRC32: Not defined\n");
    #endif

    #ifdef __ARM_FEATURE_CRYPTO
        printf("__ARM_FEATURE_CRYPTO: Defined\n");
    #else
        printf("__ARM_FEATURE_CRYPTO: Not defined\n");
    #endif

    #ifdef __ARM_FEATURE_FMA
        printf("__ARM_FEATURE_FMA: Defined\n");
    #else
        printf("__ARM_FEATURE_FMA: Not defined\n");
    #endif

    #ifdef __ARM_FEATURE_LDREX
        printf("__ARM_FEATURE_LDREX: Defined\n");
    #else
        printf("__ARM_FEATURE_LDREX: Not defined\n");
    #endif

    #ifdef __ARM_FEATURE_QBIT
        printf("__ARM_FEATURE_QBIT: Defined\n");
    #else
        printf("__ARM_FEATURE_QBIT: Not defined\n");
    #endif

    #ifdef __ARM_FEATURE_SAT
        printf("__ARM_FEATURE_SAT: Defined\n");
    #else
        printf("__ARM_FEATURE_SAT: Not defined\n");
    #endif

    #ifdef __ARM_FEATURE_SIMD32
        printf("__ARM_FEATURE_SIMD32: Defined\n");
    #else
        printf("__ARM_FEATURE_SIMD32: Not defined\n");
    #endif

    #ifdef __ARM_FEATURE_NUMERIC_MAXMIN
        printf("__ARM_FEATURE_NUMERIC_MAXMIN: Defined\n");
    #else
        printf("__ARM_FEATURE_NUMERIC_MAXMIN: Not defined\n");
    #endif

    #ifdef __ARM_FEATURE_DIRECTED_ROUNDING
        printf("__ARM_FEATURE_DIRECTED_ROUNDING: Defined\n");
    #else
        printf("__ARM_FEATURE_DIRECTED_ROUNDING: Not defined\n");
    #endif

    #ifdef __ARM_FEATURE_MATMUL_INT8
        printf("__ARM_FEATURE_MATMUL_INT8: Defined\n");
    #else
        printf("__ARM_FEATURE_MATMUL_INT8: Not defined\n");
    #endif

    #ifdef __ARM_FEATURE_BF16
        printf("__ARM_FEATURE_BF16: Defined\n");
    #else
        printf("__ARM_FEATURE_BF16: Not defined\n");
    #endif

    // NEON and vector extensions
    #ifdef __ARM_NEON
        printf("__ARM_NEON: Defined\n");
    #else
        printf("__ARM_NEON: Not defined\n");
    #endif

    #ifdef __ARM_FEATURE_SVE
        printf("__ARM_FEATURE_SVE: Defined\n");
    #else
        printf("__ARM_FEATURE_SVE: Not defined\n");
    #endif

    return 0;
}
