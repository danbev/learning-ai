## Advanced Matrix Extension
This is an extension for Intel x86 instruction set which was introduces in 2020.

Traditional SIMD (AVX, SVE, NEON) works on 1D arrays of data, vectors. But AMX
works with 2d registers (tiles).

There is a specific hardware  unit that performs the matrix operations called
the Tile Matrix Multiply Unit (TMUL).
When we call `_tile_dpbssd()` we are sending a command to the TMUL unit to
perform all those matrix multiply-accumulate operations.

Registers:
* tmm0-tmm7: Tile registers (up to 1KB each)

Traditional: AVX-512 has 512-bit (64-byte) vector registers:
```
zmm0: [a0|a1|a2|a3|a4|a5|a6|a7|...] (1D, 16 int32s)
```

AMX: Has 8 tile registers (tmm0-tmm7), each up to 1KB
```
tmm0:  [a00 a01 a02 a03]   (2D, 16 rows × 64 bytes max)
       [a10 a11 a12 a13]
       [a20 a21 a22 a23]
       [a30 a31 a32 a33]
```

Each tile can be configured to different sizes (up to 16 rows × 64 bytes).
So 64 bytes per row means we can store 16 32-bit ints.

Unlike other extensions where registers have fixed sizes, AMX tiles are
configurable:
```
tilecfg.rows[0] = 4;     // tmm0 has 4 rows
tilecfg.colsb[0] = 16;   // tmm0 has 16 bytes per row
```
