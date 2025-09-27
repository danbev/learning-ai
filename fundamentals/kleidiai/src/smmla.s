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
    
    // This is just to show how the addition part of smmla works, here we are setting all
    // values of v2 to 10, so the result of the multiplication will be added to 10.
    //movi    v2.4s, #10

    // One instruction does 2x8 × 8x2 → 2x2
    // This computes: C[2x2] = A[2x8] × B[8x2] + C[2x2] (accumulate)
    smmla   v2.4s, v0.16b, v1.16b
    // destination iv v2 and it will be 4 int32_t values (2x2 matrix)
    // source1 is v0 16 bytes (2x8 matrix of int8_t)
    // source2 is v1 16 bytes (8x2 matrix of int8_t)
    
    // Store result back to memory for inspection
    adrp    x2, result_matrix@PAGE
    add     x2, x2, result_matrix@PAGEOFF
    st1     {v2.4s}, [x2]
    
    ldp     x29, x30, [sp], #16
    ret

.data
.p2align 4  // Align to 16 bytes

// Matrix A: 2x8 matrix of int8_t
// A = [1, 2, 3, 4, 5, 6, 7, 8]      <- Row 0
//     [9,10,11,12,13,14,15,16]      <- Row 1
matrix_a:
    .byte   1, 2, 3, 4, 5, 6, 7, 8
    .byte   9,10,11,12,13,14,15,16

// Matrix B: 8x2 matrix of int8_t  
// B = [1, 2]
//     [3, 4]
//     [5, 6]
//     [7, 8]
//     [9,10]
//     [1, 2]
//     [3, 4]
//     [5, 6]
matrix_b:
    .byte  1, 3, 5, 7,  9, 1, 3, 5 // column 0
    .byte  2, 4, 6, 8, 10, 2, 4, 6 // column 1

// Result matrix: 2x2 matrix of int32_t (results from SMMLA)
.p2align 4  // Align to 16 bytes
result_matrix:
    .word   0, 0, 0, 0
