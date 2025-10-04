## ARM Scalable Vector Extension (SVE)
This is ARM's newer more advanced SIMD technology which was introduced in 2016.
It can have variable length vectors from 128 bits to 2048.

Designed for high-performance computing (HPC) and servers

Now, it is not a replacement for NEON, its an additional extension. NEON is universally
available on ARM64, while SVE is only available on some high-end CPUs.

SVE intrinsics use `sv` (scalar vector) prefix.

### SVE2 (2019)
Enhancements to SVE, introduced in 2019. Adds more instructions useful for
general-purpose computing.
