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
