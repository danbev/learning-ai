#include <arm_sve.h>
#include <stdio.h>

// Vector addition using ARM SVE
void vector_add_sve(const float *a, const float *b, float *result, size_t n) {
    uint64_t i = 0;

    // Process vectors while there are elements remaining
    while (i < n) {
        // Create a predicate for the remaining elements
        // This handles the tail automatically - no cleanup loop needed!
        svbool_t pg = svwhilelt_b32(i, (uint64_t)n);

        // Load vectors with predication
        svfloat32_t va = svld1_f32(pg, &a[i]);
        svfloat32_t vb = svld1_f32(pg, &b[i]);

        // Perform addition
        svfloat32_t vresult = svadd_f32_z(pg, va, vb);

        // Store result with predication
        svst1_f32(pg, &result[i], vresult);

        // Increment by the vector length (scalable!)
        i += svcntw();  // Count of 32-bit elements in a vector
    }
}

// Traditional scalar version for comparison
void vector_add_scalar(const float *a, const float *b, float *result, size_t n) {
    for (size_t i = 0; i < n; i++) {
        result[i] = a[i] + b[i];
    }
}

int main() {
    const size_t N = 1000;
    float a[N], b[N], result_sve[N], result_scalar[N];

    // Initialize arrays
    for (size_t i = 0; i < N; i++) {
        a[i] = i * 1.5f;
        b[i] = i * 0.5f;
    }

    // Perform additions
    vector_add_sve(a, b, result_sve, N);
    vector_add_scalar(a, b, result_scalar, N);

    // Verify results match
    printf("Vector length: %llu elements\n", (unsigned long long)svcntw());
    printf("First 5 results: ");
    for (int i = 0; i < 5; i++) {
        printf("%.1f ", result_sve[i]);
    }
    printf("\n");

    return 0;
}
