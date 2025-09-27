#include "kai_lhs_quant_pack_qai8dxp_f32.h"
#include "kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0.h"
#include "kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm.h"

#include <stdio.h>
#include <float.h>

int main() {
    const size_t m = 13;
    const size_t n = 17;
    const size_t k = 18;
    
    const size_t lhs_native_size_f32   = m * k * sizeof(float);
    const size_t rhs_native_size_qs4cx = n * (k / 2) * sizeof(uint8_t);
    const size_t dst_size_f32          = m * n * sizeof(float);
    
    uint8_t * lhs_native_mtx_f32   = new uint8_t[lhs_native_size_f32];
    uint8_t * rhs_native_mtx_qs4cx = new uint8_t[rhs_native_size_qs4cx];
    uint8_t * dst_mtx_f32          = new uint8_t[dst_size_f32];
    
    const size_t rhs_scales_size_f32 = n * sizeof(float);
    uint8_t* rhs_scales_f32 = new uint8_t[rhs_scales_size_f32];
    
    // Initialize LHS matrix (A) with some test values
    float* lhs_as_float = (float*)lhs_native_mtx_f32;
    for (size_t i = 0; i < m * k; i++) {
        lhs_as_float[i] = 1.0f + (float)(i % 5);  // Values 1.0 to 5.0
    }
    
    // Initialize RHS matrix (B) with 4-bit quantized values
    // For 4-bit, each byte contains two 4-bit values
    for (size_t i = 0; i < n * (k / 2); i++) {
        rhs_native_mtx_qs4cx[i] = 0x23;  // Two 4-bit values: 2 and 3
    }
    
    float* scales_as_float = (float*)rhs_scales_f32;
    for (size_t i = 0; i < n; i++) {
        scales_as_float[i] = 1.0f;  // Scale factor of 1.0
    }
    
    printf("Initialized matrices with test data\n");
    printf("LHS (A): %zu x %zu, first few values: ", m, k);
    for (int i = 0; i < 5 && i < m * k; i++) {
        printf("%.1f ", lhs_as_float[i]);
    }
    printf("\n");
    
    const size_t mr = kai_get_mr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm();
    const size_t nr = kai_get_nr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm();
    const size_t kr = kai_get_kr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm();
    const size_t sr = kai_get_sr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm();
    
    printf("Block sizes - mr: %zu, nr: %zu, kr: %zu, sr: %zu\n", mr, nr, kr, sr);
    
    const size_t lhs_packed_size = kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(m, k, mr, kr, sr);
    const size_t rhs_packed_size = kai_get_rhs_packed_size_rhs_pack_nxk_qsi4cxp_qs4cxs1s0(n, k, nr, kr, sr);
    
    uint8_t* lhs_packed_mtx_qa8dx = new uint8_t[lhs_packed_size];
    uint8_t* rhs_packed_mtx_qs4cx = new uint8_t[rhs_packed_size];
    
    struct kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0_params params;
    params.lhs_zero_point = 1;
    params.rhs_zero_point = 8;
    
    printf("Packing RHS matrix...\n");
    kai_run_rhs_pack_nxk_qsi4cxp_qs4cxs1s0(
        1, n, k, nr, kr, sr,
        (const uint8_t*)(rhs_native_mtx_qs4cx),
        NULL,                                   // No bias
        (const float*)(rhs_scales_f32),
        rhs_packed_mtx_qs4cx,
        0, &params);
    
    printf("Packing and quantizing LHS matrix...\n");
    kai_run_lhs_quant_pack_qai8dxp_f32(
        m, k, mr, kr, sr, 0,
        (const float*)lhs_native_mtx_f32,
        k * sizeof(float),
        lhs_packed_mtx_qa8dx);
    
    const size_t dst_stride = n * sizeof(float);
    
    printf("Running matrix multiplication...\n");
    kai_run_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm(
        m, n, k,
        (const void*)lhs_packed_mtx_qa8dx,
        (const void*)rhs_packed_mtx_qs4cx,
        (float*)dst_mtx_f32,
        dst_stride,
        sizeof(float),
        -FLT_MAX, FLT_MAX);
    
    printf("\nResult matrix (%zu x %zu):\n", m, n);
    float* result = (float*)dst_mtx_f32;
    
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
    
    // Check if we got non-zero results
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
    
    delete[] lhs_native_mtx_f32;
    delete[] rhs_native_mtx_qs4cx;
    delete[] dst_mtx_f32;
    delete[] rhs_scales_f32;
    delete[] lhs_packed_mtx_qa8dx;
    delete[] rhs_packed_mtx_qs4cx;
    
    return 0;
}
