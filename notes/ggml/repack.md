## Repack
Lets say we have 4 quantized blocks and each one contains a delta (scale factor)
and 16 quantized values:
```
Block0: delta=2.5, quants=[ 1, 2, 3, 4,  5, 6, 7, 8, 9, 10,11,12, 13,14,15,16]
Block1: delta=1.8, quants=[17,18,19,20, 21,22,23,24, 25,26,27,28, 29,30,31,32]
Block2: delta=3.1, quants=[33,34,35,36, 37,38,39,40, 41,42,43,44, 45,46,47,48]
Block3: delta=2.2, quants=[49,50,51,52, 53,54,55,56, 57,58,59,60, 61,62,63,64]
```
Repacked:
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
If we did not to specify constexpr the other branches might be in the compiled
and not eliminated completely. Though modern compilers will probably optimize
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
So this is still using 4 bits per quantized value but we are going to combine
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
```
Input matrix (4 rows × k columns):
Row 0: [f f f f f f f f ... ] (k elements)
Row 1: [f f f f f f f f ... ] (k elements)
Row 2: [f f f f f f f f ... ] (k elements)
Row 3: [f f f f f f f f ... ] (k elements)
       └─ QK8_0=32 ─┘└─ 32 ─┘ (k/32 blocks per row)
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
calculate the delta (scale factor), actually this is the inverse os the scale
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
Next, we have the actual packing part of the current block, here we iterate from
0 to QK8_0*4 (128).

_wip_

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
