#include <arm_sve.h>
#include <stdio.h>

// Vector addition using ARM SVE
void vector_add_sve(const float *a, const float *b, float *result, size_t n) {
    uint64_t i = 0;

    // Iterate over all elements in steps of vector length (8)
    while (i < n) {
        printf("Processing index: %ld\n", i);
        // Create a predicate for the remaining elements
        // The svwhilelt_b32 intrinsic generates a predicate for 32-bit elements
        // to mask off elements beyond the array bounds.
        svbool_t pg = svwhilelt_b32(i, (uint64_t) n);

        // The following is on to inspect the predicate as I was cuirious about it
        svfloat32_t test  = svdup_f32(1.0f);
        svfloat32_t zeros = svdup_f32(0.0f);
        svfloat32_t res   = svsel_f32(pg, test, zeros);

        float lanes[svcntw()];
        svst1_f32(svptrue_b32(), lanes, res);

        printf("Predicate lanes[%ld]: ", i);
        for (int j = 0; j < svcntw(); j++) {
            printf("%.0f ", lanes[j]);
        }
        printf("\n");


        // Load vectors of size svcntw() (8 32-bit elements in a vector)
        svfloat32_t va = svld1_f32(pg, &a[i]);
        svfloat32_t vb = svld1_f32(pg, &b[i]);

        // Perform addition of the vectors
        svfloat32_t vresult = svadd_f32_z(pg, va, vb);

        // Store result with predication
        svst1_f32(pg, &result[i], vresult);

        // Increment by the vector length (scalable!) and hardware dependant
        // On my Ubuntu system using QEMU, svcntw() returns 8, meaning 8 floats (32-bit) per vector
        // SO each register can store 8 floats, 8*32 = 256
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
    const size_t N = 1005;
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
