.section .data
    .align 16
    vec1: .float 1.0, 2.0, 3.0, 4.0

    .align 16
    result: .space 16

.section .note.GNU-stack,"",@progbits

.section .text
.globl main

main:
    # Load the vector into xmm0 [1.0, 2.0, 3.0, 4.0]
    movaps vec1(%rip), %xmm0

    # First we copy the original vector to xmm1
    movaps %xmm0, %xmm1        # Copy original to xmm1

    # Then we are going to shuffle/permute the elements in the original.
    # So currently the both look like this:
    # xmm0 = [1.0, 2.0, 3.0, 4.0]
    # xmm1 = [1.0, 2.0, 3.0, 4.0]
    # And a horizontal sum is summing the elements in pairs, so we move elements
    # in the original vector:
    #     [1.0, 2.0, 3.0, 4.0]
    #     [2.0, 1.0, 4.0, 3.0]
    # pos:  3    2    1    0                  
    # shufps takes an 8-bit immediate: 11100100 = 0xE4
    # This encodes _MM_SHUFFLE(2, 3, 0, 1) -> swap adjacent pairs
    # Breakdown:
    # bits 7-6=11 (pos3->2), 5-4=10 (pos2->3), 3-2=00 (pos1->0), 1-0=01 (pos0->1)
    shufps $0x4E, %xmm0, %xmm1 # xmm1 = [2.0, 1.0, 4.0, 3.0] (swapped pairs)

    # The we can add the two vectors in one operation
    addps %xmm1, %xmm0
    # xmm0 = [3.0, 3.0, 7.0, 7.0]

    # But we still don't have the horizontal sum.
    # Step 2: Shuffle to swap halves
    # shufps with 01001110 = 0x4E encodes _MM_SHUFFLE(1, 0, 3, 2) -> swap halves
    movaps %xmm0, %xmm1        # Copy pair sums to xmm1
    shufps $0x4E, %xmm0, %xmm1 # xmm1 = [7.0, 7.0, 3.0, 3.0] (swapped halves)

    # Add to get final sum in all lanes
    addps %xmm1, %xmm0         # xmm0 = [3+7, 3+7, 7+3, 7+3] = [10.0, 10.0, 10.0, 10.0]

    # Store the result (all lanes contain 10.0)
    movaps %xmm0, result(%rip)

    # The sum is now in the first element: 1.0 + 2.0 + 3.0 + 4.0 = 10.0
    
    xorq %rax, %rax
    ret
