// Left-hand side (LHS) packing:
#include "kai_lhs_quant_pack_qai8dxp_f32.h"
// Right-hand side (RHS) packing:
#include "kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0.h"
// Matrix multiplication:
#include "kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm.h"
// op = matmul
// activation = clamp
// output type = f32
// input matrix A type  = qai8dxp4x8
// input matrix B type  = qsi4cxp8x8
// block sizes = 8x8x32
// engine = neon
// cpu feature = , i8mm

// LHS: Converts float32 → quantized asymmetric int8 + packs for optimal memory layout
// RHS: Handles 4-bit quantized data packing with scales and zero points  
// Matmul: The actual SMMLA-optimized matrix multiplication micro-kernel

#include <stdio.h>
#include <float.h>

int main() {
    const size_t m = 13; // output rows
    const size_t n = 17; // output columns
    const size_t k = 18; // reduction dimension (LHS columns = RHS rows)
    // Deliberately chose non-power-of-2 sizes to test edge case handling
    // Result will be 13×17 matrix from (13×18) × (18×17)
    
    const size_t matrix_a_size      = m * k * sizeof(float);
    const size_t matrix_b_size      = n * (k / 2) * sizeof(uint8_t);
    const size_t matrix_result_size = m * n * sizeof(float);
    
    uint8_t* matrix_a      = new uint8_t[matrix_a_size];
    uint8_t* matrix_b      = new uint8_t[matrix_b_size];
    uint8_t* matrix_result = new uint8_t[matrix_result_size];
    
    const size_t scales_size = n * sizeof(float);
    uint8_t* scales = new uint8_t[scales_size];
    
    // Initialize matrix a with some test values
    float* matrix_a_as_float = (float*) matrix_a;
    for (size_t i = 0; i < m * k; i++) {
        matrix_a_as_float[i] = 1.0f + (float)(i % 5);  // Values 1.0 to 5.0
    }
    
    // Initialize matrix b with 4-bit quantized values
    // For 4-bit, each byte contains two 4-bit values
    for (size_t i = 0; i < n * (k / 2); i++) {
        matrix_b[i] = 0x23;  // Two 4-bit values: 2 and 3
    }
    
    float* scales_as_float = (float*)scales;
    for (size_t i = 0; i < n; i++) {
        scales_as_float[i] = 1.0f;  // Scale factor of 1.0
    }
    
    printf("Initialized matrices with test data\n");
    printf("matrix A: %zu x %zu, first few values: ", m, k);
    for (int i = 0; i < 5 && i < m * k; i++) {
        printf("%.1f ", matrix_a_as_float[i]);
    }
    printf("\n");
    
    // Number of rows in matrix A
    const size_t mr = kai_get_mr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm();
    // Number of columns in matrix B
    const size_t nr = kai_get_nr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm();
    // Number of columns in blocks for the reduction dimension
    const size_t kr = kai_get_kr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm();
    // How many elements share the same scale factor along the reduction dimension 
    const size_t sr = kai_get_sr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm();
    printf("Block sizes - mr: %zu, nr: %zu, kr: %zu, sr: %zu\n", mr, nr, kr, sr);
    
    const size_t matrix_a_packed_size = kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(m,
            k, mr, kr, sr);

    const size_t matrix_b_packed_size = kai_get_rhs_packed_size_rhs_pack_nxk_qsi4cxp_qs4cxs1s0(n,
            k, nr, kr, sr);
    
    uint8_t* lhs_packed_mtx_qa8dx = new uint8_t[matrix_a_packed_size];
    uint8_t* rhs_packed_mtx_qs4cx = new uint8_t[matrix_b_packed_size];
    
    struct kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0_params params;
    params.lhs_zero_point = 1;
    params.rhs_zero_point = 8;
    
    printf("Packing matrix B...\n");
    kai_run_rhs_pack_nxk_qsi4cxp_qs4cxs1s0(1, n, k, nr, kr, sr,
        (const uint8_t*)(matrix_b),
        NULL,                                   // No bias
        (const float*)(scales),
        rhs_packed_mtx_qs4cx,
        0, &params);
    
    printf("Packing and quantizing matrix A...\n");
    kai_run_lhs_quant_pack_qai8dxp_f32(
        m, k, mr, kr, sr, 0,
        (const float*)matrix_a,
        k * sizeof(float),
        lhs_packed_mtx_qa8dx);
    
    const size_t dst_stride = n * sizeof(float);
    
    printf("Running matmul operation...\n");
    kai_run_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm(
        m, n, k,
        (const void*)lhs_packed_mtx_qa8dx,
        (const void*)rhs_packed_mtx_qs4cx,
        (float*)matrix_result,
        dst_stride,
        sizeof(float),
        -FLT_MAX, // min clamp value
        FLT_MAX); // max clamp value
    
    printf("\nResult matrix (%zu x %zu):\n", m, n);
    float* result = (float*) matrix_result;
    
    // Print just the first few rows and columns to keep output manageable
    size_t print_rows = (m > 5) ? 5 : m;
    size_t print_cols = (n > 8) ? 8 : n;
    
    for (size_t i = 0; i < print_rows; i++) {
        for (size_t j = 0; j < print_cols; j++) {
            printf("%8.2f ", result[i * n + j]);
        }
        if (n > print_cols) printf("...");
        printf("\n");
    }
    if (m > print_rows) printf("...\n");
    
    bool has_nonzero = false;
    for (size_t i = 0; i < m * n; i++) {
        if (result[i] != 0.0f) {
            has_nonzero = true;
            break;
        }
    }
    
    if (has_nonzero) {
        printf("\n✓ SUCCESS: Got non-zero results!\n");
    } else {
        printf("\n✗ Still getting all zeros - check quantization parameters\n");
    }
    
    delete[] matrix_a;
    delete[] matrix_b;
    delete[] matrix_result;
    delete[] scales;
    delete[] lhs_packed_mtx_qa8dx;
    delete[] rhs_packed_mtx_qs4cx;
    
    return 0;
}
