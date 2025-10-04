## ARM NEON (SIMD)
NEON is ARM's SIMD (Single Instruction, Multiple Data) architecture extension.


### Registers
NEON has 32 registers (V0-V31), each 128 bits wide. These can be accessed in multiple
ways.  Similar to x86 (where rax, eax, ax, al are views of the same register), ARM's `q`
and `v` registers are the same physical registers with different naming conventions.
```
Physical Register 0 (128 bits total):
┌─────────────────────────────────────────────────────────────┐
│                         q0 (128 bits)                       │
│                       "Quad-word view"                      │
└─────────────────────────────────────────────────────────────┘
│◄────────────────── 128 bits = 16 bytes ────────────────────►│

Can also be viewed as:
┌──────────────────────────────┬──────────────────────────────┐
│        d0 (64 bits)          │        d1 (64 bits)          │
│     "Double-word view"       │     "Double-word view"       │
└──────────────────────────────┴──────────────────────────────┘

Or as:
┌───────────┬───────────┬───────────┬───────────┐
│s0 (32bit) │s1 (32bit) │s2 (32bit) │s3 (32bit) │
│         "Single-word view" (scalars)          │
└───────────┴───────────┴───────────┴───────────┘

Or as:
┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│ h0  │ h1  │ h2  │ h3  │ h4  │ h5  │ h6  │ h7  │
│     "Half-word view" (16-bit each)      │     │
└─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘

Or as:
┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
│b0│b1│b2│b3│b4│b5│b6│b7│b8│b9│ba│bb│bc│bd│be│bf│
│         "Byte view" (8-bit each)  │  │  │  │  │
└──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘
```
