.section .data
    # In this case we only have the vectors to add and the result.
    # So this will have proper alignment which is required for `movaps`.
    # But this is fragile and if we add more data, we might get a segfault.
    some_byte: .byte 42

    .align 16
    vec1: .float 1.0, 2.0, 3.0, 4.0
    
    .align 16
    vec2: .float 5.0, 6.0, 7.0, 8.0

    .align 16
    result: .space 16                  # uninitialized space for the result
    #result: .space 16, 0              # initialized to 0
    #result: .float 0.0, 0.0, 0.0, 0.0 # initialized to 0

# make stack nonexecutable ("") 
.section .note.GNU-stack,"",@progbits


.section .text
.globl main

main:
    # Load vectors and add them
    movaps vec1(%rip), %xmm0
    movaps vec2(%rip), %xmm1
    addps %xmm1, %xmm0
    
    # Store result. movaps requires 16-byte alignment
    movaps %xmm0, result(%rip)
    
    # Return 0
    xorq %rax, %rax
    ret
