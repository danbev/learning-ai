#include <stdio.h>
#include <sys/sysctl.h>
#include <sys/types.h>
#include <string.h>
#include <stdbool.h>

void print_sysctl_string(const char* name) {
    char buffer[256];
    size_t size = sizeof(buffer);
    if (sysctlbyname(name, buffer, &size, NULL, 0) == 0) {
        printf("%-35s: %s\n", name, buffer);
    } else {
        printf("%-35s: Not available\n", name);
    }
}

void print_sysctl_int(const char* name) {
    int value;
    size_t size = sizeof(value);
    if (sysctlbyname(name, &value, &size, NULL, 0) == 0) {
        printf("%-35s: %d\n", name, value);
    } else {
        printf("%-35s: Not available\n", name);
    }
}

void print_sysctl_uint64(const char* name) {
    uint64_t value;
    size_t size = sizeof(value);
    if (sysctlbyname(name, &value, &size, NULL, 0) == 0) {
        printf("%-35s: %llu\n", name, value);
    } else {
        printf("%-35s: Not available\n", name);
    }
}

void check_arm_features() {
    printf("\nARM Feature Detection (Compile-time)\n");
#ifdef __aarch64__
    printf("Architecture                    : ARM64 (AArch64) ✓\n");
#else
    printf("Architecture                    : Not ARM64\n");
    return;
#endif
    
#ifdef __ARM_NEON
    printf("NEON SIMD                      : ✓\n");
#else
    printf("NEON SIMD                      : ✗\n");
#endif
    
#ifdef __ARM_FEATURE_DOTPROD
    printf("Dot Product (SDOT)             : ✓\n");
#else
    printf("Dot Product (SDOT)             : ✗\n");
#endif
    
#ifdef __ARM_FEATURE_MATMUL_INT8
    printf("i8mm (Matrix Multiply)         : ✓\n");
#else
    printf("i8mm (Matrix Multiply)         : ✗\n");
#endif
    
#ifdef __ARM_FEATURE_BF16
    printf("BF16 (Brain Float)             : ✓\n");
#else
    printf("BF16 (Brain Float)             : ✗\n");
#endif
    
#ifdef __ARM_FEATURE_SVE
    printf("SVE (Scalable Vector Ext)      : ✓\n");
#else
    printf("SVE (Scalable Vector Ext)      : ✗\n");
#endif
    
#ifdef __ARM_FEATURE_SME
    printf("SME (Scalable Matrix Ext)      : ✓\n");
#else
    printf("SME (Scalable Matrix Ext)      : ✗\n");
#endif
}

int main() {
    printf("Apple Silicon M3 CPU info\n");
    
    // Basic CPU info
    print_sysctl_string("machdep.cpu.brand_string");
    print_sysctl_string("hw.model");
    print_sysctl_string("hw.machine");
    
    printf("\nCPU Counts:\n");
    print_sysctl_int("hw.ncpu");
    print_sysctl_int("hw.activecpu"); 
    print_sysctl_int("hw.physicalcpu");
    print_sysctl_int("hw.logicalcpu");
    print_sysctl_int("hw.perflevel0.physicalcpu");  // Performance cores
    print_sysctl_int("hw.perflevel1.physicalcpu");  // Efficiency cores
    
    // Compile-time ARM feature detection
    check_arm_features();
    
    return 0;
}
