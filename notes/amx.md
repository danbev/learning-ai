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
Each value in this matrix is a 4-byte DWORD. This becomes important when
populating the matrix and we need to take this into account.
```c++
    int8_t A[4][16] = {0};
    A[0][0] = 1;  A[0][4] = 2;  A[0][8] = 3;   A[0][12] = 4;
```
Notice that each value starts at index: 0, 4, 8, 12, were I initially thought
I could just do 0, 1, 2, 3. This is because each value is a 4-byte DWORD.

Each tile can be configured to different sizes (up to 16 rows × 64 bytes).
So 64 bytes per row means we can store 16 32-bit ints.

Unlike other extensions where registers have fixed sizes, AMX tiles are
configurable:
```
tilecfg.rows[0] = 4;     // tmm0 has 4 rows
tilecfg.colsb[0] = 16;   // tmm0 has 16 bytes per row
```

### Tile Matrix Multiply Unit (TMUL)
This unit preforms dot-product SIMD operations where the unit of operation
is 4 bytes, 32 bits. To perform an operation on INT8 data it groups 4 ints in
a 32-bit integer, and performs an operation that outputs a 32-bit integer.
TMUL processes 4-byte DWORDS at a time.
For each C[i,j] element it will perform:
```
C[i][j] = sum of dot products of 4-byte groups from A[i] and B[j]
```
```
// Extract 4 bytes from DWORD k in row i of A
a0 = A[i][k*4 + 0]
a1 = A[i][k*4 + 1]
a2 = A[i][k*4 + 2]
a3 = A[i][k*4 + 3]

// Extract 4 bytes from DWORD k in column j of B
b0 = B[k][j*4 + 0]
b1 = B[k][j*4 + 1]
b2 = B[k][j*4 + 2]
b3 = B[k][j*4 + 3]

// Compute dot product
C[i][j] += a0*b0 + a1*b1 + a2*b2 + a3*b3
```
