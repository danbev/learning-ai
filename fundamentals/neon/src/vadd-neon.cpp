#include <arm_neon.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void vector_add_neon(const float* a, const float* b, float* result, size_t n) {
    size_t i = 0;

    // Process 4 floats at a time (128-bit NEON registers)
    for (; i + 4 <= n; i += 4) {
        // Load 4 floats from each array
        float32x4_t va = vld1q_f32(&a[i]);
        float32x4_t vb = vld1q_f32(&b[i]);

        // Perform addition
        float32x4_t vresult = vaddq_f32(va, vb);

        // Store result
        vst1q_f32(&result[i], vresult);
    }

    // Handle remaining elements (scalar tail)
    for (; i < n; i++) {
        result[i] = a[i] + b[i];
    }
}

// This version will actually use NEON as the comopiler will auto-vectorize it
/*
The following is one iteration of the loop in assembly language (ARM64):
LBB1_11:
	ldp	q0, q1, [x10, #-32]   // Load 2 128-bit quad vectors (8 floats) (a?)
	ldp	q2, q3, [x10], #64    // Load 2 more and post-increment pointer
	ldp	q4, q5, [x11, #-32]   // Load 2 128-bit quad vectors from second array (b?)
	ldp	q6, q7, [x11], #64    // Load 2 more and post-increment pointer
	fadd.4s	v0, v0, v4        // NEON fadd instruction
	fadd.4s	v1, v1, v5
	fadd.4s	v2, v2, v6
	fadd.4s	v3, v3, v7
	stp	q0, q1, [x9, #-32]
	stp	q2, q3, [x9], #64
	subs	x12, x12, #16
	b.ne	LBB1_11
void vector_add_scalar(const float* a, const float* b, float* result, size_t n) {
    for (size_t i = 0; i < n; i++) {
        result[i] = a[i] + b[i];
    }
}
*/

// Truly scalar version - use volatile to prevent vectorization
void vector_add_scalar(const float* a, const float* b, float* result, size_t n) {
    for (size_t i = 0; i < n; i++) {
        // The volatile prevents the compiler from optimizing this
        volatile float temp_a = a[i];
        volatile float temp_b = b[i];
        result[i] = temp_a + temp_b;
    }
}

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

int main() {
    const size_t N = 10000000;  // 10 million elements

    // Allocate aligned memory for better NEON performance
    float* a             = (float*)aligned_alloc(16, N * sizeof(float));
    float* b             = (float*)aligned_alloc(16, N * sizeof(float));
    float* result_neon   = (float*)aligned_alloc(16, N * sizeof(float));
    float* result_scalar = (float*)aligned_alloc(16, N * sizeof(float));

    // Initialize arrays
    for (size_t i = 0; i < N; i++) {
        a[i] = i * 1.5f;
        b[i] = i * 0.5f;
    }

    double start = get_time();
    vector_add_neon(a, b, result_neon, N);
    double neon_time = get_time() - start;

    start = get_time();
    vector_add_scalar(a, b, result_scalar, N);
    double scalar_time = get_time() - start;

    int errors = 0;
    for (size_t i = 0; i < N && errors < 5; i++) {
        if (result_neon[i] != result_scalar[i]) {
            printf("Mismatch at %zu: NEON=%.2f, Scalar=%.2f\n", i, result_neon[i], result_scalar[i]);
            errors++;
        }
    }

    if (errors == 0) {
        printf("✓ Results verified - all %zu elements match!\n\n", N);
    }

    // Print performance results
    printf("Performance Results:\n");
    printf("Array size: %zu elements (%.1f MB)\n", N, N * sizeof(float) / 1e6);
    printf("NEON time:   %.4f seconds\n", neon_time);
    printf("Scalar time: %.4f seconds\n", scalar_time);
    printf("Speedup:     %.2fx\n", scalar_time / neon_time);
    printf("\nFirst 5 results: ");
    for (int i = 0; i < 5; i++) {
        printf("%.1f ", result_neon[i]);
    }
    printf("\n");

    // Cleanup
    free(a);
    free(b);
    free(result_neon);
    free(result_scalar);

    return 0;
}
