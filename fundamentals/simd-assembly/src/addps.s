.section .data
    #.align 16
    vec1: .float 1.0, 2.0, 3.0, 4.0
    #.align 16
    vec2: .float 5.0, 6.0, 7.0, 8.0
    #.align 16
    result: .space 16

# make stack nonexecutable ("") 
.section .note.GNU-stack,"",@progbits


.section .text
.globl main

main:
    # Load vectors and add them
    movaps vec1(%rip), %xmm0
    movaps vec2(%rip), %xmm1
    addps %xmm1, %xmm0
    
    # Store result
    movaps %xmm0, result(%rip)
    
    # Return 0
    xorq %rax, %rax
    ret
