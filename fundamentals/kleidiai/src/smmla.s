.text
.global _main
.p2align 4  // Align to 16 bytes

// i8mm is an instruction set extension and contains a number of instructions for
// matrix // multiplication using int8 data types. i8 for 8-bit integers, mm for
// matrix multiply.
// 
// The instruction used in this example is Signed Matrix Multiply-Accumulate (SMMLA).

// Reminder or ARM assembly syntax:
// instruction destination, source1, source2
// For example:
// mov x29, sp        // x29 = sp
// add x0, x1, x2     // x0 = x1 + x2
_main:
    // Set up stack frame
    // Store pair of registers (frame pointer and link register)
    // x29 is the frame pointer register, and x30 is the link register (the return address)
    // We first calculate the address (sp - 16) to make room for 16 bytes on the stack
    // (2 registers of 8 bytes each). Then we store both registers at that address,
    // and finally sp is updated to point to the new location. The '!' means
    // pre-decrement: calculate address, store, then update sp.
    stp     x29, x30, [sp, #-16]!
    // [sp, #-16]! is equivalent to: *(sp -16) in C.

    // So the instruction is just setting the current framepointer to sp which is
    // the framepointer for this function, and before this address on the stack we
    // have the return value and the callers frame pointer.
    mov     x29, sp
    
    // Initialize our test matrices. Branch with link, so return address is saved in lr (x30).
    bl      setup_matrices
    
    bl      do_smmla
    
    // inspect results in memory
    // Set return value
    mov     w0, #0              // Return 0
    
    // Restore stack and return
    ldp     x29, x30, [sp], #16
    ret

// Setup test matrices for 2x8 × 8x2 → 2x2 multiplication
setup_matrices:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    
    // Load address of our data section
    adrp    x0, matrix_a@PAGE
    add     x0, x0, matrix_a@PAGEOFF
    
    adrp    x1, matrix_b@PAGE  
    add     x1, x1, matrix_b@PAGEOFF
    
    // Load Matrix A (2x8 of int8_t) into v0 register
    // This loads 16 bytes: [a00,a01,a02,a03,a04,a05,a06,a07,a10,a11,a12,a13,a14,a15,a16,a17]
    ld1     {v0.16b}, [x0]
    
    // Load Matrix B (8x2 of int8_t) into v1 register  
    // This loads 16 bytes: [b00,b01,b10,b11,b20,b21,b30,b31,b40,b41,b50,b51,b60,b61,b70,b71]
    ld1     {v1.16b}, [x1]
    
    // Initialize result matrix C to zero
    eor     v2.16b, v2.16b, v2.16b  // Clear v2 (will hold 2x2 result as 4 int32s)
    
    ldp     x29, x30, [sp], #16
    ret

do_smmla:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    
    // One instruction does 2x8 × 8x2 → 2x2
    // SMMLA Vd.4S, Vn.16B, Vm.16B
    // This computes: C[2x2] = A[2x8] × B[8x2] + C[2x2] (accumulate)
    smmla   v2.4s, v0.16b, v1.16b
    
    // At this point:
    // v2.s[0] = A[0][0]*B[0][0] + A[0][1]*B[1][0] + ... + A[0][7]*B[7][0] = C[0][0]
    // v2.s[1] = A[0][0]*B[0][1] + A[0][1]*B[1][1] + ... + A[0][7]*B[7][1] = C[0][1]  
    // v2.s[2] = A[1][0]*B[0][0] + A[1][1]*B[1][0] + ... + A[1][7]*B[7][0] = C[1][0]
    // v2.s[3] = A[1][0]*B[0][1] + A[1][1]*B[1][1] + ... + A[1][7]*B[7][1] = C[1][1]
    
    // Store result back to memory for inspection
    adrp    x2, result_matrix@PAGE
    add     x2, x2, result_matrix@PAGEOFF
    st1     {v2.4s}, [x2]
    
    ldp     x29, x30, [sp], #16
    ret

// Data section - our test matrices
.data
.p2align 4  // Align to 16 bytes

// Matrix A: 2x8 matrix of int8_t
// A = [1, 2, 3, 4, 5, 6, 7, 8]      <- Row 0
//     [9,10,11,12,13,14,15,16]      <- Row 1
matrix_a:
    .byte   1, 2, 3, 4, 5, 6, 7, 8     // Row 0: A[0][0] through A[0][7]
    .byte   9,10,11,12,13,14,15,16     // Row 1: A[1][0] through A[1][7]

// Matrix B: 8x2 matrix of int8_t  
// B = [1, 2]     <- Row 0: B[0][0], B[0][1]
//     [3, 4]     <- Row 1: B[1][0], B[1][1]
//     [5, 6]     <- Row 2: B[2][0], B[2][1]
//     [7, 8]     <- Row 3: B[3][0], B[3][1]
//     [9,10]     <- Row 4: B[4][0], B[4][1]
//     [1, 2]     <- Row 5: B[5][0], B[5][1]
//     [3, 4]     <- Row 6: B[6][0], B[6][1] 
//     [5, 6]     <- Row 7: B[7][0], B[7][1]
matrix_b:
    .byte   1, 2, 3, 4, 5, 6, 7, 8     // B[0][0],B[0][1],B[1][0],B[1][1],B[2][0],B[2][1],B[3][0],B[3][1]
    .byte   9,10, 1, 2, 3, 4, 5, 6     // B[4][0],B[4][1],B[5][0],B[5][1],B[6][0],B[6][1],B[7][0],B[7][1]

// Result matrix: 2x2 matrix of int32_t (results from SMMLA)
.p2align 4  // Align to 16 bytes
result_matrix:
    .word   0, 0, 0, 0                 // C[0][0], C[0][1], C[1][0], C[1][1]

// Expected results for verification:
// C[0][0] = 1*1 + 2*3 + 3*5 + 4*7 + 5*9 + 6*1 + 7*3 + 8*5 = 204
// C[0][1] = 1*2 + 2*4 + 3*6 + 4*8 + 5*10 + 6*2 + 7*4 + 8*6 = 236  
// C[1][0] = 9*1 + 10*3 + 11*5 + 12*7 + 13*9 + 14*1 + 15*3 + 16*5 = 620
// C[1][1] = 9*2 + 10*4 + 11*6 + 12*8 + 13*10 + 14*2 + 15*4 + 16*6 = 716
