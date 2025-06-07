// This is the equivalent of the x86_64 addsp instruction example
.section __DATA,__data
    .align 4
    vec1: .float 1.0, 2.0, 3.0, 4.0
    .align 4
    vec2: .float 5.0, 6.0, 7.0, 8.0
    .align 4
    result: .space 16

.section __TEXT,__text
.globl _main
.align 4

_main:
    // Load vectors using NEON SIMD instructions
    adrp x0, vec1@PAGE         // Get page address of vec1
    add x0, x0, vec1@PAGEOFF   // Add page offset to get full address
    ld1 {v0.4s}, [x0]          // Load 4 floats from vec1 into v0

    adrp x1, vec2@PAGE         // Get page address of vec2
    add x1, x1, vec2@PAGEOFF   // Add page offset to get full address
    ld1 {v1.4s}, [x1]          // Load 4 floats from vec2 into v1

    fadd v0.4s, v0.4s, v1.4s   // Add 4 floats in parallel: v0 += v1

    adrp x2, result@PAGE       // Get page address of result
    add x2, x2, result@PAGEOFF // Add page offset to get full address
    st1 {v0.4s}, [x2]                          // Store v0 to result array

    // Return 0 (success)
    mov w0, #0                 // Set return value to 0
    ret                        // Return to caller
