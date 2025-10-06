#include <immintrin.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>

int main(int argc, char **argv) {
    printf("AMX Matrix Multiplication: C = A * B\n");
    printf("Computing 4x4 = 4x4 * 4x4 using int8\n\n");
    
    typedef struct __tile_config {
        uint8_t  palette_id;
        uint8_t  start_row;
        uint8_t  reserved_0[14];
        uint16_t colsb[16];
        uint8_t  rows[16];
    } __attribute__((packed)) __tilecfg;
    
    __tilecfg tilecfg;
    memset(&tilecfg, 0, sizeof(tilecfg));
    
    tilecfg.palette_id = 1;
    tilecfg.start_row = 0;
    
    // For 4x4 result, we need:
    // A: 4 rows x 16 bytes (holds 4 values, each replicated 4 times as required by AMX)
    // B: 4 rows x 16 bytes (each row contributes to one output column)
    // C: 4 rows x 16 bytes (holds 4 int32 results per row)
    
    tilecfg.colsb[0] = 16;  // A: 16 bytes per row
    tilecfg.rows[0] = 4;
    
    tilecfg.colsb[1] = 16;  // B: 16 bytes per row
    tilecfg.rows[1] = 4;
    
    tilecfg.colsb[2] = 16;  // C: 16 bytes per row (4 int32 values)
    tilecfg.rows[2] = 4;
    
    _tile_loadconfig(&tilecfg);
    
    // Define simple 4x4 matrices for demonstration
    // Matrix A (stored with 4-byte groups as AMX requires)
    int8_t A[4][16] = {0};
    // Row 0: [1, 2, 3, 4] - each value in a 4-byte group
    A[0][0] = 1; A[0][1] = 0; A[0][2]  = 0; A[0][3]  = 0;
    A[0][4] = 2; A[0][5] = 0; A[0][6]  = 0; A[0][7]  = 0;
    A[0][8] = 3; A[0][9] = 0; A[0][10] = 0; A[0][11] = 0;
    A[0][12]= 4; A[0][13]= 0; A[0][14] = 0; A[0][15] = 0;
    
    // Row 1: [5, 6, 7, 8]
    A[1][0] = 5; A[1][4] = 6; A[1][8] = 7; A[1][12] = 8;
    
    // Row 2: [1, 1, 1, 1]
    A[2][0] = 1; A[2][4] = 1; A[2][8] = 1; A[2][12] = 1;
    
    // Row 3: [2, 2, 2, 2]
    A[3][0] = 2; A[3][4] = 2; A[3][8] = 2; A[3][12] = 2;
    
    // Matrix B (identity matrix for simple verification)
    // Each row k of B contributes to column k of the output
    int8_t B[4][16] = {0};
    B[0][0]  = 1;  // Row 0 -> contributes to output column 0
    B[1][4]  = 1;  // Row 1 -> contributes to output column 1
    B[2][8]  = 1;  // Row 2 -> contributes to output column 2
    B[3][12] = 1;  // Row 3 -> contributes to output column 3
    
    // Result matrix
    int32_t C[4][4] = {0};
    
    // Load A into ttm0
    _tile_loadd(0, A, 16);
    // Load B into ttm1
    _tile_loadd(1, B, 16);
    // Initialize C tile to zero
    _tile_zero(2);

    // one instruction to do the matrix multiply and accumulate
    // tmm2 += tmm0 * tmm1
    _tile_dpbssd(2, 0, 1);

    // Store result from ttm2 to C
    _tile_stored(2, C, 16);
    _tile_release();
    
    // Display matrices
    printf("Matrix A (4x4):\n");
    printf("  1  2  3  4\n");
    printf("  5  6  7  8\n");
    printf("  1  1  1  1\n");
    printf("  2  2  2  2\n\n");
    
    printf("Matrix B (4x4 identity):\n");
    printf("  1  0  0  0\n");
    printf("  0  1  0  0\n");
    printf("  0  0  1  0\n");
    printf("  0  0  0  1\n\n");
    
    printf("Result C = A * B (should equal A):\n");
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%3d ", C[i][j]);
        }
        printf("\n");
    }
    
    int correct = 1;
    int8_t expected[4][4] = {{1,2,3,4}, {5,6,7,8}, {1,1,1,1}, {2,2,2,2}};
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            if (C[i][j] != expected[i][j]) {
                correct = 0;
                break;
            }
        }
    }
    
    printf("\n%s\n", correct ? "Result is correct!" : "Result mismatch");
    
    return 0;
}
