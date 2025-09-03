## Repack
This is a feature that is specific to the CPU backend and is about data layout
and how to optimize the layout for matrix operations.

### What is repack?
So [quantization](quantization.md) time rearrangement makes sure that within
a block the values are spaced so that values in a column can be loaded together
in one simd operation. Repacking allows us to something similar but for
multiple blocks which would otherwise be layed out one after another.

Lets say we have 4 quantized blocks and each one contains a delta (scale factor)
and 16 quantized values:
```
Block0: delta=2.5, quants=[ 1, 2, 3, 4,  5, 6, 7, 8, 9, 10,11,12, 13,14,15,16]
Block1: delta=1.8, quants=[17,18,19,20, 21,22,23,24, 25,26,27,28, 29,30,31,32]
Block2: delta=3.1, quants=[33,34,35,36, 37,38,39,40, 41,42,43,44, 45,46,47,48]
Block3: delta=2.2, quants=[49,50,51,52, 53,54,55,56, 57,58,59,60, 61,62,63,64]
```
After repacking these block they would look like this:
```
Deltas: [2.5, 1.8, 3.1, 2.2]

Quants: [ 1, 2, 3, 4] [17,18,19,20]  [33,34,35,36]  [49,50,51,52]     <- chunk 0 from each block
        [ 5, 6, 7, 8] [21,22,23,24]  [37,38,39,40]  [53,54,55,56]     <- chunk 1 from each block  
        [9,10,11, 12] [25,26,27,28]  [41,42,43,44]  [57,58,59,60]     <- chunk 2 from each block
        [13,14,15,16] [29,30,31,32]  [45,46,47,48]  [61,62,63,64]     <- chunk 3 from each block
```
When performing matrix multiplication there is often a need to multiply elements
from multiple blocks simultaneously. Without repacking/rearranging the data,
the CPU would have to load block0[0], block1[0], block2[0], block3[0] which would
result in cache misses and more memory accesses which leads to performance
degradation.

By repacking the data, we can load all 4 elements by loading one SIMD register
(128 bits) = [Block0_chunk0, Block1_chunk0, Block2_chunk0, Block3_chunk0].
```
load xmm0, [repacked_addr]   # Gets chunk from all 4 blocks at once
```

So, lets take `QK4_0` as an example just to understand this better:
```c++
#define QK4_0 32
typedef struct {
    ggml_half d;           // delta
    uint8_t qs[QK4_0 / 2]; // nibbles / quants
} block_q4_0;
```
This is saying that a `block_q4_0` can store 32 quantized values and we will
store them using 4 bits each (nibble):
```c++
    uint8_t qs[16];
```
```
   [ nibble0  ][ nibble1   ]
   +--+--+--+--+--+--+--+--+
0  |0 |1 |2 |3 |4 |5 |6 |7 |     (0 1)
   +--+--+--+--+--+--+--+--+     
1  |0 |1 |2 |3 |4 |5 |6 |7 |     (2 3)
   +--+--+--+--+--+--+--+--+
2  |0 |1 |2 |3 |4 |5 |6 |7 |     (4 5)
   +--+--+--+--+--+--+--+--+
3  |0 |1 |2 |3 |4 |5 |6 |7 |     (6 7)
   +--+--+--+--+--+--+--+--+
4  |0 |1 |2 |3 |4 |5 |6 |7 |     (8 9)
   +--+--+--+--+--+--+--+--+
5  |0 |1 |2 |3 |4 |5 |6 |7 |     (10 11)
   +--+--+--+--+--+--+--+--+ 
6  |0 |1 |2 |3 |4 |5 |6 |7 |     (12 13)
   +--+--+--+--+--+--+--+--+
7  |0 |1 |2 |3 |4 |5 |6 |7 |     (14 15)
   +--+--+--+--+--+--+--+--+
8  |0 |1 |2 |3 |4 |5 |6 |7 |     (16 17)
   +--+--+--+--+--+--+--+--+
9  |0 |1 |2 |3 |4 |5 |6 |7 |     (18 19)
   +--+--+--+--+--+--+--+--+
10 |0 |1 |2 |3 |4 |5 |6 |7 |     (20 21)
   +--+--+--+--+--+--+--+--+
11 |0 |1 |2 |3 |4 |5 |6 |7 |     (22 23)
   +--+--+--+--+--+--+--+--+
12 |0 |1 |2 |3 |4 |5 |6 |7 |     (24 25)
   +--+--+--+--+--+--+--+--+
13 |0 |1 |2 |3 |4 |5 |6 |7 |     (26 27)
   +--+--+--+--+--+--+--+--+
14 |0 |1 |2 |3 |4 |5 |6 |7 |     (28 29)
   +--+--+--+--+--+--+--+--+
15 |0 |1 |2 |3 |4 |5 |6 |7 |     (30 31)
   +--+--+--+--+--+--+--+--+
   [ nibble0  ][ nibble1   ]
```
So `QK4_0 32` means that we can store 32 quantized values in a `block_q4_0`.
So each original float32 value becomes a 4-bit integer.

### block_q4_0x4
In repack `block_q4_0x4` means we take 4 separate `block_q4_0` blocks like the
one above which gives use 4 scale/delta values, and 4*32 = 128 quantized values.

This is declared with a using statement:
```c++
                        The bit width of the quantization (so 4 bits for each value)
                           ↓
using block_q4_0x4 = block<4, 4>;
                              ↑
                              Number of of blocks to combine (4 in this case)
```
And block is defined as follows:
```c++
template <int K, int N> struct block {
    ggml_half d[N];
    int8_t    qs[(QK_0<K>() * N * K) / 8];
};
```
Again, `K` is the bit width of the quantization (4 bits in this case) and `N`
is the number of blocks to combine (4 in this case).
And notice that this is using `QK_0<K>()`:
```c++
template <int K> constexpr int QK_0() {
    if constexpr (K == 4) {
        return QK4_0;
    }
    if constexpr (K == 8) {
        return QK8_0;
    }
    return -1;
}
```
Notice the usage of `if constexpr` which is nice because this enables the compiler
to avoid generating any of other branches that are not used, so if we use `K=4`
it will only generate:
```c++
    if constexpr (K == 4) {
        return QK4_0;
    }
```
If we did not to specify constexpr the other branches might be in the compilation
unit and not eliminated completely. Though modern compilers will probably optimize
this anyway using constexpr, it is still a good practice to signal the intent.

So for our case this will return `QK4_0` which is 32.
```c++
    int8_t    qs[(QK_0<K>() * N * K) / 8];
    int8_t    qs[        32 * 4 * 4) / 8];
    int8_t    qs[               512) / 8]; 
    int8_t    qs[64]; // 64 bytes

template <int K, int N> struct block {
    ggml_half d[4];
    int8_t    qs[64];  // 4*16 bytes.
};
```
So in the case of a single `block_q4_0` we had one delta/scale value and
16 `uint8_t` quantized values, and since we only use 4 bits per value we can
therefor store 32 quantized value in a single `block_q4_0` block.

When we repack 4 of these blocks together, we get 4 delta/scale values and
16*4=64 `uint8_t` quantized values.

We can also combine more than 4 block by specifying:
```c++
using block_q4_0x8 = block<4, 8>;
```
So this is still using 4 bits per quantized value, but we are going to combine
8 `block_q4_0` blocks together which gives us 8 delta/scale values and
32*8=256 `uint8_t` quantized values.

There are also:
```c++
using block_q8_0x4 = block<8, 4>;
using block_q8_0x8 = block<8, 8>;
```

The `block_q8_0` is defined as follows:
```c++
#define QK8_0 32
typedef struct {
    ggml_half d;       // delta
    int8_t  qs[QK8_0]; // quants
} block_q8_0;
```
Notice that this used `int8_t` instead of `uint8_t` for the quantized values and
also the qs arrays is of size 32 and not 16 because each quantized values now
requires 8-bits and not 4. And there is the choice of packing 4 of these blocks
or 8 together.


### macros
```c++
GGML_CPU_NATIVE_IMPL(ggml_quantize_mat_q8_0_4x4)
````
This macro can be found in `ggml-cpu-impl.h`:
```c++
#define GGML_CPU_NATIVE_IMPL(name) GGML_WEAK_ALIAS(name, name ## _generic)
```
And if we look at the GCC version of `GGML_WEAK_ALIAS` we find:
```c++
# define GGML_WEAK_ALIAS(name, alias) GGML_DO_PRAGMA(weak name = alias) // NOLINT
```
So the above will expand to:
```c++
#pragma weak ggml_quantize_mat_q8_0_4x4 = ggml_quantize_mat_q8_0_4x4_generic
```
How this works is that if the symbol `ggml_quantize_mat_q8_0_4x4` is not defined
then the compiler will set the symbol to point to
`ggml_quantize_mat_q8_0_4x4_generic`.

### ggml_quantize_mat_q8_0_4x4_generic
```c++
void ggml_quantize_mat_q8_0_4x4_generic(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
```
Here `x` is the input, `vy` is the output and the `v` probably stands for `void`
pointer. k is the width (number of columns) of the matrix.
```c++
    assert(QK8_0 == 32);
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;
```
So `k` is the number of columns and it has to be a multiple of `QK8_0` which is
32. So it could be 32, 64, 96, 128, etc. `nb` is number of blocks per row. 

For example, lets say k=32:
```
Input matrix (4 rows × k columns):

x = [
  Row 0: [f00 f01 f02 f03 f04 f05 f06 f07 ... f28 f29 f30 f31]  (32 floats)
  Row 1: [f10 f11 f12 f13 f14 f15 f16 f17 ... f38 f39 f40 f41]  (32 floats)
  Row 2: [f20 f21 f22 f23 f24 f25 f26 f27 ... f48 f49 f50 f51]  (32 floats)
  Row 3: [f30 f31 f32 f33 f34 f35 f36 f37 ... f58 f59 f60 f61]  (32 floats)
]
total size: 4 rows * 32 columns * sizeof(float) = 4 * 32 * 4 = 512 bytes
```

Next, `vy` is casted to `block_q8_0x4` pointer:
```c++
    block_q8_0x4 * GGML_RESTRICT y = (block_q8_0x4 *) vy;
```
```c++
    // scalar
    const int blck_size_interleave = 4;
    float srcv[4][QK8_0];
    float id[4];
```
Here `srcv` is a 2D array that will hold the values for each of the 4 rows and
id is the array that will hold the deltas/scale factors for each of the 4 rows.

Following that we will iterate over each block, and for each block we have to
calculate the delta (scale factor), actually this is the inverse of the scale
factor (1.0f / d) to avoid division, and do to that we need to find the maxium
value in the block. And since we are extracting values from x these are also
stored in `srcv` which can then be used in the next loop where we calculate
the actual quantized values.
```c++
    for (int i = 0; i < nb; i++) {
        for (int row_iter = 0; row_iter < 4; row_iter++) {
            float amax = 0.0f; // absolute max

            for (int j = 0; j < QK8_0; j++) {
                srcv[row_iter][j] = x[row_iter * k + i * QK8_0 + j];
                amax = MAX(amax, fabsf(srcv[row_iter][j]));
            }

            const float d = amax / ((1 << 7) - 1);
            id[row_iter] = d ? 1.0f / d : 0.0f;

            y[i].d[row_iter] = GGML_FP32_TO_FP16(d);
        }
```
So `srcv` will be filled in the above loop with something like this:
```
srcv[0][0-31] = [f00 f01 f02 f03 f04 f05 f06 f07 ... f28 f29 f30 f31]
srcv[1][0-31] = [f10 f11 f12 f13 f14 f15 f16 f17 ... f38 f39 f40 f41]
srcv[2][0-31] = [f20 f21 f22 f23 f24 f25 f26 f27 ... f48 f49 f50 f51]
srcv[3][0-31] = [f30 f31 f32 f33 f34 f35 f36 f37 ... f58 f59 f60 f61]
```

Next, we have the actual "re-packing" part of the current block, here we iterate
from 0 to QK8_0*4 (128).

```c++
        for (int j = 0; j < QK8_0 * 4; j++) {
            int src_offset = (j / (4 * blck_size_interleave)) * blck_size_interleave;
            int src_id = (j % (4 * blck_size_interleave)) / blck_size_interleave;
            src_offset += (j % blck_size_interleave);

            float x0 = srcv[src_id][src_offset] * id[src_id];
            y[i].qs[j] = roundf(x0);
        }
    }
}
GGML_CPU_NATIVE_IMPL(ggml_quantize_mat_q8_0_4x4)
```
```
int src_offset = (j / (4 * blck_size_interleave)) * blck_size_interleave;
int src_offset = (j / (4 * 4                   )) * 4;

j = 0  -> (0  / 16) * 4 = 0 * 4 = 0
j = 1  -> (1  / 16) * 4 = 0 * 4 = 0
j = 2  -> (2  / 16) * 4 = 0 * 4 = 0
...
j = 15 -> (15 / 16) * 4 = 0 * 4 = 0
j = 16 -> (16 / 16) * 4 = 1 * 4 = 4
j = 17 -> (17 / 16) * 4 = 1 * 4 = 4
j = 18 -> (18 / 16) * 4 = 1 * 4 = 4
...
j = 31 -> (31 / 16) * 4 = 1 * 4 = 4
j = 32 -> (32 / 16) * 4 = 2 * 4 = 8
j = 33 -> (33 / 16) * 4 = 2 * 4 = 8
...
j = 47 -> (47 / 16) * 4 = 2 * 4 = 8
j = 48 -> (48 / 16) * 4 = 3 * 4 = 12
j = 49 -> (49 / 16) * 4 = 3 * 4 = 12
...
j = 63 -> (63 / 16) * 4 = 3 * 4 = 12
j = 64 -> (64 / 16) * 4 = 4 * 4 = 16
j = 65 -> (65 / 16) * 4 = 4 * 4 = 16
...
j = 79 -> (79 / 16) * 4 = 4 * 4 = 16
j = 80 -> (80 / 16) * 4 = 5 * 4 = 20
j = 81 -> (81 / 16) * 4 = 5 * 4 = 20
...
j = 95 -> (95 / 16) * 4 = 5 * 4 = 20
j = 96 -> (96 / 16) * 4 = 6 * 4 = 24
j = 97 -> (97 / 16) * 4 = 6 * 4 = 24
j = 98 -> (98 / 16) * 4 = 6 * 4 = 24
...
j = 111 -> (111 / 16) * 4 = 6 * 4 = 24
j = 112 -> (112 / 16) * 4 = 7 * 4 = 28
j = 113 -> (113 / 16) * 4 = 7 * 4 = 28
...
j = 127 -> (127 / 16) * 4 = 7 * 4 = 28
```
So this will give us 8 groups of 16 values.

And notice that this is added to as well:
```
src_offset += (j % blck_size_interleave);

0 % 4 = 0
1 % 4 = 1
2 % 4 = 2
3 % 4 = 3
4 % 4 = 0
5 % 4 = 1
6 % 4 = 2
7 % 4 = 3
8 % 4 = 0
...
```

Then we have the source id which is the row of the `srcv` array
that we are currently working with:
```c++
int src_id = (j % (4 * blck_size_interleave)) / blck_size_interleave;
int src_id = (0 % (4 *                    4)) / 4

j =  0  -> (0 % 16) / 4 =  0 / 4 = 0
j =  1  -> (1 % 16) / 4 =  1 / 4 = 0
j =  2  -> (2 % 16) / 4 =  2 / 4 = 0
j =  3  -> (3 % 16) / 4 =  3 / 4 = 0
j =  4  -> (4 % 16) / 4 =  4 / 4 = 1
j =  5  -> (5 % 16) / 4 =  5 / 4 = 1
j =  6  -> (6 % 16) / 4 =  6 / 4 = 1
j =  7  -> (7 % 16) / 4 =  7 / 4 = 1
j =  8  -> (8 % 16) / 4 =  8 / 4 = 2
j =  9  -> (9 % 16) / 4 =  9 / 4 = 2
j = 10 -> (10 % 16) / 4 = 10 / 4 = 2
j = 11 -> (11 % 16) / 4 = 11 / 4 = 2
j = 12 -> (12 % 16) / 4 = 12 / 4 = 3
j = 13 -> (13 % 16) / 4 = 13 / 4 = 3
j = 14 -> (14 % 16) / 4 = 14 / 4 = 3
j = 15 -> (15 % 16) / 4 = 15 / 4 = 3
j = 16 -> (16 % 16) / 4 =  0 / 4 = 0
j = 17 -> (17 % 16) / 4 =  1 / 4 = 0
j = 18 -> (18 % 16) / 4 =  2 / 4 = 0
j = 19 -> (19 % 16) / 4 =  3 / 4 = 0
j = 20 -> (20 % 16) / 4 =  4 / 4 = 1
j = 21 -> (21 % 16) / 4 =  5 / 4 = 1
j = 22 -> (22 % 16) / 4 =  6 / 4 = 1
j = 23 -> (23 % 16) / 4 =  7 / 4 = 1
j = 24 -> (24 % 16) / 4 =  8 / 4 = 2
j = 25 -> (25 % 16) / 4 =  9 / 4 = 2
j = 26 -> (26 % 16) / 4 = 10 / 4 = 2
...
```

```
       src_offset 
         |
  src_id |
     ↓   ↓
srcv[0][0-31] = [f00 f01 f02 f03 f04 f05 f06 f07 ... f28 f29 f30 f31]
srcv[1][0-31] = [f10 f11 f12 f13 f14 f15 f16 f17 ... f38 f39 f40 f41]
srcv[2][0-31] = [f20 f21 f22 f23 f24 f25 f26 f27 ... f48 f49 f50 f51]
srcv[3][0-31] = [f30 f31 f32 f33 f34 f35 f36 f37 ... f58 f59 f60 f61]
```
And the values are still float32 at this stage, they have not been quantized yet.
This happens in the loop just below:
```c++
float x0 = srcv[src_id][src_offset] * id[src_id];
```
And this is using the `id` array which contains the inverse deltas/scale factors
so we can just multiply the float32 value with the inverse delta to get the
quantized value in the range of -127 to 127 (since we are using 8 bits for the
quantized values). And this is stored in 
```c++
y[i].qs[j] = roundf(x0);
```
And notice that the second loop if from 0-128, which makes sense if we think
about it; the quantized values are just stored in an 1d array `qs`, so they all
come after each other, but with the nice property that they quantized values
from each row are after each other making simd operations more efficient.

So instead of having the order we had above we now have the following order:
```
y[0].qs[0-3]   = [f00, f01, f02, f03]   <- first 4 values from first row
y[0].qs[4-7]   = [f10, f11, f12, f13]   <- first 4 values from second row
y[0].qs[8-11]  = [f20, f21, f22, f23]   <- first 4 values from third row
y[0].qs[12-15] = [f30, f31, f32, f33]   <- first 4 values from fourth row
y[0].qs[16-19] = [f04, f05, f06, f07]   <- second 4 values from first row
y[0].qs[20-23] = [f14, f15, f16, f17]   <- second 4 values from second row
y[0].qs[24-27] = [f24, f25, f26, f27]   <- second 4 values from third row
y[0].qs[28-31] = [f34, f35, f36, f37]   <- second 4 values from fourth row
y[0].qs[32-35] = [f08, f09, f10, f11]   <- third 4 values from first row
y[0].qs[36-39] = [f18, f19, f20, f21]   <- third 4 values from second row
y[0].qs[40-43] = [f28, f29, f30, f31]   <- third 4 values from third row
y[0].qs[44-47] = [f38, f39, f40, f41]   <- third 4 values from fourth row
y[0].qs[48-51] = [f16, f17, f18, f19]   <- fourth 4 values from first row
y[0].qs[52-55] = [f22, f23, f24, f25]   <- fourth 4 values from second row
y[0].qs[56-59] = [f32, f33, f34, f35]   <- fourth 4 values from third row
y[0].qs[60-63] = [f42, f43, f44, f45]   <- fourth 4 values from fourth row
y[0].qs[64-67] = [f20, f21, f22, f23]   <- fifth 4 values from first row
y[0].qs[68-71] = [f26, f27, f28, f29]   <- fifth 4 values from second row
y[0].qs[72-75] = [f36, f37, f38, f39]   <- fifth 4 values from third row
y[0].qs[76-79] = [f46, f47, f48, f49]   <- fifth 4 values from fourth row
y[0].qs[80-83] = [f24, f25, f26, f27]   <- sixth 4 values from first row
y[0].qs[84-87] = [f30, f31, f32, f33]   <- sixth 4 values from second row
y[0].qs[88-91] = [f40, f41, f42, f43]   <- sixth 4 values from third row
y[0].qs[92-95] = [f50, f51, f52, f53]   <- sixth 4 values from fourth row
y[0].qs[96-99] = [f32, f33, f34, f35]   <- seventh 4 values from first row
y[0].qs[100-103] = [f38, f39, f40, f41] <- seventh 4 values from second row
y[0].qs[104-107] = [f48, f49, f50, f51] <- seventh 4 values from third row
y[0].qs[108-111] = [f58, f59, f60, f61] <- seventh 4 values from fourth row
y[0].qs[112-115] = [f44, f45, f46, f47] <- eighth 4 values from first row
y[0].qs[116-119] = [f50, f51, f52, f53] <- eighth 4 values from second row    
y[0].qs[120-123] = [f60, f61, f62, f63] <- eighth 4 values from third row
y[0].qs[124-127] = [f70, f71, f72, f73] <- eighth 4 values from fourth row
```
So we for y[0] it will have 64 entries and each group of 4 are 8-bit quantized
values for float32 values.

For a simd matrix operations this is ideel:
```c++
__m128i b0 = _mm_set1_epi32(vector[0]);  // [b0, b0, b0, b0] (as 32-bit)
__m128i b1 = _mm_set1_epi32(vector[1]);  // [b1, b1, b1, b1]
__m128i b2 = _mm_set1_epi32(vector[2]);  // [b2, b2, b2, b2]
__m128i b3 = _mm_set1_epi32(vector[3]);  // [b3, b3, b3, b3]

// Load matrix columns (each load gets one element from each row)
__m128i col0 = _mm_loadu_si128((__m128i*)&interleaved_matrix[0]);   // [a00,a10,a20,a30]
__m128i col1 = _mm_loadu_si128((__m128i*)&interleaved_matrix[4]);   // [a01,a11,a21,a31]
__m128i col2 = _mm_loadu_si128((__m128i*)&interleaved_matrix[8]);   // [a02,a12,a22,a32]
__m128i col3 = _mm_loadu_si128((__m128i*)&interleaved_matrix[12]);  // [a03,a13,a23,a33]

// Multiply and accumulate (4 dot products in parallel)
__m128i prod0 = _mm_mullo_epi32(col0, b0);  // [a00×b0, a10×b0, a20×b0, a30×b0]
__m128i prod1 = _mm_mullo_epi32(col1, b1);  // [a01×b1, a11×b1, a21×b1, a31×b1]
__m128i prod2 = _mm_mullo_epi32(col2, b2);  // [a02×b2, a12×b2, a22×b2, a32×b2]
__m128i prod3 = _mm_mullo_epi32(col3, b3);  // [a03×b3, a13×b3, a23×b3, a33×b3]

// Sum up all products
__m128i sum01 = _mm_add_epi32(prod0, prod1);
__m128i sum23 = _mm_add_epi32(prod2, prod3);
__m128i final_sum = _mm_add_epi32(sum01, sum23);

// Store result: [c0, c1, c2, c3]
_mm_storeu_si128((__m128i*)result, final_sum);
```

### Usage in ggml
In ggml/CMakeLists.txt we have the following option:
```console
option(GGML_CPU_REPACK       "ggml: use runtime weight conversion of Q4_0 to Q4_X_X" ON)
```
And in `ggml/src/ggml-cpu/CMakeLists.txt` we have:
```console
    if (GGML_CPU_REPACK)
        target_compile_definitions(${GGML_CPU_NAME} PRIVATE GGML_USE_CPU_REPACK)
    endif()
```
And if we look where the macro `GGML_USE_CPU_REPACK` is used we find it in
`ggml-cpu.cpp`:
```c++
std::vector<ggml_backend_buffer_type_t> & ggml_backend_cpu_get_extra_buffer_types() {
    static std::vector<ggml_backend_buffer_type_t> bufts = []() {
        std::vector<ggml_backend_buffer_type_t> bufts;

#if defined(__AMX_INT8__) && defined(__AVX512VNNI__)
        if (ggml_backend_amx_buffer_type()) {
            bufts.push_back(ggml_backend_amx_buffer_type());
        }
#endif

#ifdef GGML_USE_CPU_KLEIDIAI
        if (ggml_backend_cpu_kleidiai_buffer_type()) {
            bufts.push_back(ggml_backend_cpu_kleidiai_buffer_type());
        }
#endif

#ifdef GGML_USE_CPU_REPACK
        if (ggml_backend_cpu_repack_buffer_type()) {
            bufts.push_back(ggml_backend_cpu_repack_buffer_type());
        }
#endif

        return bufts;
    }();

    return bufts;
}
```
And it is also used in :
```c++
static ggml_backend_feature * ggml_backend_cpu_get_features(ggml_backend_reg_t reg) {
    static std::vector<ggml_backend_feature> features = []() {
        ggml_cpu_init();

        ...
    #ifdef GGML_USE_CPU_REPACK
        features.push_back({ "REPACK", "1" });
    #endif
```
