#include <arm_sve.h>
#include <stdio.h>
#include <sys/prctl.h>
#include <stdlib.h>

// Macros from GGML for SVE
#define GGML_F32_VEC              svfloat32_t
#define GGML_F32_VEC_SET1         svdup_f32
#define GGML_F32_VEC_LOAD(p)      svld1_f32(svptrue_b32(), p)
#define GGML_F32_VEC_STORE(p, a)  svst1_f32(svptrue_b32(), p, a)
#define GGML_F32_VEC_MUL(a, b)    svmul_f32_x(svptrue_b32(), a, b)

// Get SVE vector length in bytes
static inline int ggml_cpu_get_sve_cnt(void) {
    static int sve_cnt = 0;
    if (sve_cnt == 0) {
        sve_cnt = PR_SVE_VL_LEN_MASK & prctl(PR_SVE_GET_VL);
    }
    return sve_cnt;
}

// The GGML function with the bug
void ggml_vec_scale_f32(const int n, float * y, const float v) {
    const int sve_register_length = ggml_cpu_get_sve_cnt() * 8;
    const int ggml_f32_epr = sve_register_length / 32;
    const int ggml_f32_step = 2 * ggml_f32_epr;
    
    GGML_F32_VEC vx = GGML_F32_VEC_SET1(v);
    const int np = (n & ~(ggml_f32_step - 1));
    
    svfloat32_t ay1, ay2;
    
    // Main loop, will process 16 elements
    int n_processed = 0;
    for (int i = 0; i < np; i += ggml_f32_step) {
        ay1 = GGML_F32_VEC_LOAD(y + i);
        ay1 = GGML_F32_VEC_MUL(ay1, vx);
        GGML_F32_VEC_STORE(y + i, ay1);
        
        ay2 = GGML_F32_VEC_LOAD(y + i + 1*ggml_f32_epr);
        ay2 = GGML_F32_VEC_MUL(ay2, vx);
        GGML_F32_VEC_STORE(y + i + 1*ggml_f32_epr, ay2);
        n_processed += ggml_f32_step;
    }
    printf("Processed %d elements in main loop\n", n_processed);
    
    printf("Leftover elements to process: %d\n", n - np);
    // 16 is less than 25 so we have leftovers
    /*
    if (np < n) {
        printf("Create predicate with %d, %d\n", np, n);
        // 1 1 1 1 1 1 1 1
        // So the first 8 lanes are active, but recall that we a have 9
        // leftovers.
        svbool_t pg = svwhilelt_b32(np, n);

        // Just printing the predicate lanes to see what they are
        svfloat32_t test  = svdup_f32(1.0f);
        svfloat32_t zeros = svdup_f32(0.0f);
        svfloat32_t res   = svsel_f32(pg, test, zeros);

        float lanes[svcntw()];
        svst1_f32(svptrue_b32(), lanes, res);

        printf("Predicate lanes:\n");
        for (int j = 0; j < svcntw(); j++) {
            printf("%.0f ", lanes[j]);
        }
        printf("\n");

        // load with predicate, and y+np points to element 16 so this will
        // load elements 16,17,18,19,20,21,22,23
        ay1 = svld1_f32(pg, y + np);
        // Then we multiply with the predicate, so only the first 8 lanes so
        // this will multiply elements 16,17,18,19,20,21,22,23
        ay1 = svmul_f32_m(pg, ay1, vx);
        // then we store with the predicate, so only the first 8 lanes
        svst1_f32(pg, y + np, ay1);
    }
    */

    // To fix the issue above we need an nother iteration
    for (int i = np; i < n; i += ggml_f32_epr) {
        svbool_t pg = svwhilelt_b32(i, n);
        svfloat32_t test  = svdup_f32(1.0f);
        svfloat32_t zeros = svdup_f32(0.0f);
        svfloat32_t res   = svsel_f32(pg, test, zeros);

        float lanes[svcntw()];
        svst1_f32(svptrue_b32(), lanes, res);

        printf("Predicate lanes:\n");
        for (int j = 0; j < svcntw(); j++) {
            printf("%.0f ", lanes[j]);
        }
        printf("\n");



        ay1 = svld1_f32(pg, y + i);
        ay1 = svmul_f32_m(pg, ay1, vx);
        svst1_f32(pg, y + i, ay1);
    }
}

int main() {
    printf("=== GGML vec_scale Leftover Bug Demo ===\n\n");
    
    int sve_cnt = ggml_cpu_get_sve_cnt();
    // elements per register
    int ggml_f32_epr = (sve_cnt * 8) / 32;
    // we process two registers per loop iteration
    int ggml_f32_step = 2 * ggml_f32_epr;
    
    printf("SVE vector length: %d bytes (%d bits)\n", sve_cnt, sve_cnt * 8);
    printf("Elements per register (epr): %d\n", ggml_f32_epr);
    printf("Step size (two registers per loop): %d\n\n", ggml_f32_step);
    
    // Issue to reproduce : step_size + more_than_epr leftovers
    const int n = ggml_f32_step + ggml_f32_epr + 1;  // 16 + 8 + 1 = 25
    const float scale = 2.0f;
    
    printf("Testing with n=%d, scale=%.1f\n", n, scale);
    printf("Expected: np=%d, leftovers=%d\n", ggml_f32_step, n - ggml_f32_step);
    
    float *y = (float *)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) {
        y[i] = (float)i;
    }
    
    printf("Before scaling:\n");
    for (int i = 0; i < n; i++) {
        printf("y[%2d] = %5.1f\n", i, y[i]);
    }
    
    ggml_vec_scale_f32(n, y, scale);
    
    printf("\nAfter scaling:\n");
    for (int i = 0; i < n; i++) {
        float expected = (float)i * scale;
        printf("y[%2d]: %5.1f, expected: %5.1f\n", i, y[i], expected);
    }
    
    free(y);
    return 0;
}
