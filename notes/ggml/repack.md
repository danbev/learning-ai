## Repack
This is a feature that is specific to the CPU backend and is about data layout
and how to optimize the layout for matrix operations.

### What is repack?
Repack is an optimization reorganizes quantized data layout to enable efficient SIMD
operations during matrix multiplication. Instead of processing one block at a
time, it interleaves multiple blocks so that corresponding elements can be loaded
together in a single SIMD instruction.

So the [quantization](quantization.md) rearrangement makes sure that within
a block the values are spaced so that values in a column can be loaded together
in one SIMD operation.

So bare in mind that a single block_q4_0 represents a 1D array of 32 quantized
values that came from the original 32 float32 values.
```
Original data: [f0, f1, f2, f3, ..., f30, f31]  (32 float values)
                ↓ quantization
block_q4_0: {
  d = scale_factor,
  qs[16] = [packed nibbles representing all 32 values]
}
```
The block is always 32 values in a row, regardless of the original tensor
dimensions the float values came from.

Now, lets say we have a 4x64 matrix (4 rows, 64 columns):
```
Original Matrix:

Row 0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,                       ...,  62,  63]
Row 1: [64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75,             ..., 126, 127]
Row 2: [128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, ..., 190, 191]
Row 3: [192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, ..., 254, 255]
```
This will get divided into blocks of 32 consecutive values:
```
Block 0: Row 0, positions 0-31  → [0, 1, 2, 3, 4, 5, 6, 7,                 ..., 30, 31]
Block 1: Row 0, positions 32-63 → [32, 33, 34, 35, 36, 37, 38, 39,         ..., 62, 63]

Block 2: Row 1, positions 0-31  → [64, 65, 66, 67, 68, 69, 70, 71,         ..., 94, 95]
Block 3: Row 1, positions 32-63 → [96, 97, 98, 99, 100, 101, 102, 103,     ..., 126, 127]

Block 4: Row 2, positions 0-31  → [128, 129, 130, 131, 132, 133, 134, 135, ..., 158, 159]
Block 5: Row 2, positions 32-63 → [160, 161, 162, 163, 164, 165, 166, 167, ..., 190, 191]

Block 6: Row 3, positions 0-31  → [192, 193, 194, 195, 196, 197, 198, 199, ..., 222, 223]
Block 7: Row 3, positions 32-63 → [224, 225, 226, 227, 228, 229, 230, 231, ..., 254, 255]

```
So without packing, lets say we have the following matrix vector multiplication:
```
Row 0: [  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11, ...,  62,  63] [x₀]
Row 1: [ 64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75, ..., 126, 127] [x₁]
Row 2: [128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, ..., 190, 191] [x₂]
Row 3: [192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, ..., 254, 255] [x₃]
                                                                                   ...
                                                                                   [x₆₄]
result[0] = A[0][0] * x₀ + A[0][1] * x₁ + ... + A[0][63] * x₆₃
result[1] = A[1][0] * x₀ + A[1][1] * x₁ + ... + A[1][63] * x₆₃
result[2] = A[2][0] * x₀ + A[2][1] * x₁ + ... + A[2][63] * x₆₃
result[3] = A[3][0] * x₀ + A[3][1] * x₁ + ... + A[3][63] * x₆₃
```

Repacked array positions:
```
[0-3]:   [0, 64, 128, 192]     // First element from each row
[4-7]:   [1, 65, 129, 193]     // Second element from each row
[8-11]:  [2, 66, 130, 194]     // Third element from each row
[12-15]: [3, 67, 131, 195]     // Fourth element from each row
[16-19]: [4, 68, 132, 196]     // Fifth element from each row
[20-23]: [5, 69, 133, 197]     // Sixth element from each row
[24-27]: [6, 70, 134, 198]     // Seventh element from each row
[28-31]: [7, 71, 135, 199]     // Eighth element from each row
[32-35]: [8, 72, 136, 200]     // Ninth element from each row
...
[124-127]: [31, 95, 159, 223]  // Last element from each block
```

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

Just to clarify something and this might just be me and my mental model of
matrix multiplication, but I always think of matrix multiplication as the rows
in matrix A being functions, and the columns in matrix B being the arguments.
And I think of passing the input vector down through each row calling the
function. And there is nothing wrong with that way of thinking about it. But
for efficiency in practice we should also think about the other way:
```
Function wise approch:
[2 4] [2] = [2*2 + 4*3] = [4  + 12] = [16]
[5 6] [3]   [5*2 + 6*3]   [10 + 18]   [28]

Column wise approach:
[2 4] [2] = 2 [2] + 3[4] = [4]  + [12] = [16]
[5 6] [3]     [5]    [6]   [10]   [18]   [28]
```

Compute each dot product separately:
```c++
for (int row = 0; row < 4; row++) {
    result[row] = 0;
    for (int col = 0; col < 64; col++) {
        result[row] += matrix[row][col] * vector[col];
    }
}
```
But with the repacked data we can compute all 4 dot products in parallel.
This will be the column wise approach as mentioned above, so imaging something
like this:
```
  result[0] += matrix[0][j] * vector[j]
  result[1] += matrix[1][j] * vector[j] 
  result[2] += matrix[2][j] * vector[j]
  result[3] += matrix[3][j] * vector[j]
```
But instead of separate operations we can group them together:
```c++
__m128 results = _mm_setzero_ps(); // [result[0], result[1], result[2], result[3]]

for (int col = 0; col < 64; col++) {
    __m128i four_matrix_values = _mm_loadu_si128(&repacked_data[col*4]);
    // Loads [matrix[0][col], matrix[1][col], matrix[2][col], matrix[3][col]]

    __m128 broadcast_vector = _mm_set1_ps(vector[col]);
    // Creates [vector[col], vector[col], vector[col], vector[col]], because vector[j] is the same for all 4 rows

    __m128 products = _mm_mul_ps(four_matrix_values, broadcast_vector);
    // Computes [matrix[0][col] *vector[col],
    //           matrix[1][col] *vector[col],
    //           matrix[2][col] *vector[col],
    //           matrix[3][col] *vector[col]]

    results = _mm_add_ps(results, products);
    // Accumulates into [result[0], result[1], result[2], result[3]]
}
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
one above which gives us 4 scale/delta values, and 4*32 = 128 quantized values.

This is declared with a using statement:
```c++
                        The bit width of the quantization (so 4 bits for each value)
                           ↓
using block_q4_0x4 = block<4, 4>;
                              ↑
                              Number of of blocks to combine (4 in this case)
```
And a block is defined as follows:
```c++
template <int K, int N> struct block {
    ggml_half d[N];
    int8_t    qs[(QK_0<K>() * N * K) / 8];
};
```
Again, `K` is the bit width of the quantization (4 bits in this case) and `N`
is the number of blocks to combine (also 4 in this case).
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

For example, lets say k=64:
```
Input matrix (4 rows × k columns):

x = [
  Row 0: [00 f01 f02 f03 f04 f05 f06 f07 ... f60 f61 f62 f63]  (64 floats)
  Row 1: [00 f01 f02 f03 f04 f05 f06 f07 ... f60 f61 f62 f63]  (64 floats)
  Row 2: [00 f01 f02 f03 f04 f05 f06 f07 ... f60 f61 f62 f63]  (64 floats)
  Row 3: [00 f01 f02 f03 f04 f05 f06 f07 ... f60 f61 f62 f63]  (64 floats)
]
Total elements: 4 rows * 64 columns = 256 floats
total size: 256 * sizeof(float) = 1024 bytes
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
`id` is the array that will hold the deltas/scale factors for each of the 4 rows.

Following that we will iterate over each block, and for each block we have to
calculate the delta (scale factor), actually this is the inverse of the scale
factor (1.0f / d) to avoid division, and do to that we need to find the maxium
value in the block. 
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
Now, `srvc` is taking the 1d input array and populating the 2d array `srcv` so
that each row can be index using `srcv[row_iter][j]`.

Next, we have the actual "re-packing" part of the current block:
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

For a SIMD matrix operations this is ideel:
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

But returning to the `ggml_backend_cpu_get_extra_buffer_types` this is my first
time coming across this extra buffer type concept.

### Extra buffer type
So in `ggml/src/ggml-cpu/ggml-cpu.cpp` we have:

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

static ggml_backend_buffer_type_t * ggml_backend_cpu_device_get_extra_buffers_type(ggml_backend_dev_t device) {

    static std::vector<ggml_backend_buffer_type_t> extra_bufts = [] {
        std::vector<ggml_backend_buffer_type_t> bufts = ggml_backend_cpu_get_extra_buffer_types();
        bufts.push_back(nullptr);
        return bufts;
    }();

    return extra_bufts.data();

    GGML_UNUSED(device);
}
```
Now, `extra_bufts` is a static vector, and it is initalized using a
immediately-invoked lambda expression (IIFE) (notice the ()). This will call
`ggml_backend_cpu_get_extra_buffer_types` where depending on the macros that
have been defined at compile time it will add the corresponding buffer types.
Then a `nullptr` is added to the end of the vector to enable iterating over
the vector until a `nullptr` is found.

To understand when this is called we have to take a look in `src/llama.cpp`:
```c++
static int llama_model_load(const std::string & fname, std::vector<std::string> & splits, llama_model & model, llama_model_params & params) {
    ...
        if (!model.load_tensors(ml)) {
            return -2;
        }
```

Which calls into `src/llama-model.cpp`:
```c++
bool llama_model::load_tensors(llama_model_loader & ml) {
    const auto & split_mode   = params.split_mode;
    const auto & n_gpu_layers = params.n_gpu_layers;
    const auto & use_mlock    = params.use_mlock;
    const auto & tensor_split = params.tensor_split;

    const int n_layer = hparams.n_layer;

    const bool use_mmap_buffer = true;

    LLAMA_LOG_INFO("%s: loading model tensors, this can take a while... (mmap = %s)\n", __func__, ml.use_mmap ? "true" : "false");

    // build a list of buffer types for the CPU and GPU devices
    pimpl->cpu_buft_list = make_cpu_buft_list(devices, params.use_extra_bufts);
```
And this in turn calls `make_cpu_buft_list`, and notice that this is passing
`use_extra_bufts` which is set from the command line parameter
```c++
    add_opt(common_arg(
        {"-nr", "--no-repack"},
        "disable weight repacking",
        [](common_params & params) {
            params.no_extra_bufts = true;
        }
    ).set_env("LLAMA_ARG_NO_REPACK"));
```
This is later set using:
```c++
    mparams.use_extra_bufts = !params.no_extra_bufts;
```
So lets look at `make_cpu_buft_list`:
```c++
static buft_list_t make_cpu_buft_list(const std::vector<ggml_backend_dev_t> & devices, bool use_extra_bufts) {
    buft_list_t buft_list;
    ...

    // add extra buffer types
    if (use_extra_bufts) {
        auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (cpu_dev == nullptr) {
            throw std::runtime_error(format("%s: no CPU backend found", __func__));
        }

        auto * cpu_reg = ggml_backend_dev_backend_reg(cpu_dev);

        auto ggml_backend_dev_get_extra_bufts_fn = (ggml_backend_dev_get_extra_bufts_t)
            ggml_backend_reg_get_proc_address(cpu_reg, "ggml_backend_dev_get_extra_bufts");

        if (ggml_backend_dev_get_extra_bufts_fn) {
            ggml_backend_buffer_type_t * extra_bufts = ggml_backend_dev_get_extra_bufts_fn(cpu_dev);
            while (extra_bufts && *extra_bufts) {
                buft_list.emplace_back(cpu_dev, *extra_bufts);
                ++extra_bufts;
            }
        }
    }
```
Now, notice that `ggml_backend_reg_get_proc_address` takes a backend registry and
a string name and returns a void pointer:
```c++
void * ggml_backend_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_ASSERT(reg);
    if (!reg->iface.get_proc_address) {
        return NULL;
    }
    return reg->iface.get_proc_address(reg, name);
}
```
And if the interface of this backend registry has a `get_proc_address` function
then it will be called.
We can see that the passed in name is checked against a number of strings and
that they all set and return function pointers:
```c++
static void * ggml_backend_cpu_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    if (strcmp(name, "ggml_backend_set_n_threads") == 0) {
        ggml_backend_set_n_threads_t fct = ggml_backend_cpu_set_n_threads;
        return (void *)fct;
    }
    if (strcmp(name, "ggml_backend_dev_get_extra_bufts") == 0) {
        ggml_backend_dev_get_extra_bufts_t fct = ggml_backend_cpu_device_get_extra_buffers_type;
        return (void *)fct;
    }
    ...

    return NULL;

    GGML_UNUSED(reg);
}
```
So in the case we are interested this will return the function pointer to
`ggml_backend_cpu_device_get_extra_buffers_type`.
```c++
static ggml_backend_buffer_type_t * ggml_backend_cpu_device_get_extra_buffers_type(ggml_backend_dev_t device) {
    static std::vector<ggml_backend_buffer_type_t> extra_bufts = [] {
        std::vector<ggml_backend_buffer_type_t> bufts = ggml_backend_cpu_get_extra_buffer_types();
        bufts.push_back(nullptr);
        return bufts;
    }();

    return extra_bufts.data();

    GGML_UNUSED(device);
}

std::vector<ggml_backend_buffer_type_t> & ggml_backend_cpu_get_extra_buffer_types() {
    static std::vector<ggml_backend_buffer_type_t> bufts = []() {
        std::vector<ggml_backend_buffer_type_t> bufts;
        ...

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
So this will then return to:
```c++
        auto ggml_backend_dev_get_extra_bufts_fn = (ggml_backend_dev_get_extra_bufts_t)
            ggml_backend_reg_get_proc_address(cpu_reg, "ggml_backend_dev_get_extra_bufts");

--->    if (ggml_backend_dev_get_extra_bufts_fn) {
            ggml_backend_buffer_type_t * extra_bufts = ggml_backend_dev_get_extra_bufts_fn(cpu_dev);

            while (extra_bufts && *extra_bufts) {
                buft_list.emplace_back(cpu_dev, *extra_bufts);
                ++extra_bufts;
            }
        }
```
So the function pointer is checked and then called using `cpu_dev` as argument.
```console
(gdb) p extra_bufts
$5 = std::vector of length 2, capacity 2 = {
  0x7ffff5fdc3e0 <ggml_backend_cpu_repack_buffer_type()::ggml_backend_cpu_buffer_type_repack>, 0x0}

(gdb) p *extra_bufts[0]
$7 = {iface = {get_name = 0x7ffff5eb55a9 <ggml_backend_cpu_repack_buffer_type_get_name(ggml_backend_buffer_type_t)>, 
    alloc_buffer = 0x7ffff5eb55be <ggml_backend_cpu_repack_buffer_type_alloc_buffer(ggml_backend_buffer_type_t, size_t)>, 
    get_alignment = 0x7ffff5eb5643 <ggml_backend_cpu_repack_buffer_type_get_alignment(ggml_backend_buffer_type_t)>, 
    get_max_size = 0x0, get_alloc_size = 0x0, is_host = 0x0}, 
  device = 0x7ffff5fdc2c0 <ggml_backend_cpu_reg_get_device(ggml_backend_reg*, unsigned long)::ggml_backend_cpu_device>, 
  context = 0x555555dc93c0}
```
The function returs a pointer to the first entry (and that is the reasons for
adding the nullptr at the end of the vector) so they can be iterated over:
```c++
            while (extra_bufts && *extra_bufts) {
                buft_list.emplace_back(cpu_dev, *extra_bufts);
                ++extra_bufts;
            }
```
This will then be added to the `buft_list` which is of type:
```c++
using buft_list_t = std::vector<std::pair<ggml_backend_dev_t, ggml_backend_buffer_type_t>>;
```
So using emplace back we add a pair of the cpu device and the buffer type.

So the `buft_list` will now contain the repack buffer type. But when will it
be used?  
When a tensor is created in `llama_model::load_tensors` for example, for
GEMMA3 we have:
```c++

            case LLM_ARCH_GEMMA3:
            ```
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

```
In the `create_tensor` lambda we have:
```c++
auto create_tensor = [&](const LLM_TN_IMPL & tn, const std::initializer_list<int64_t> & ne, int flags) -> ggml_tensor * {
    ...
            if (!buft) {
                buft = select_weight_buft(hparams, t_meta, op, *buft_list);
                if (!buft) {
                    throw std::runtime_error(format("failed to find a compatible buffer type for tensor %s", tn.str().c_str()));
                }
            }
    ...
}

static ggml_backend_buffer_type_t select_weight_buft(const llama_hparams & hparams, ggml_tensor * tensor, ggml_op op, const buft_list_t & buft_list) {
    GGML_ASSERT(!buft_list.empty());
    for (const auto & cur : buft_list) {
        ggml_backend_dev_t cur_dev = cur.first;
        ggml_backend_buffer_type_t cur_buft = cur.second;
        if (weight_buft_supported(hparams, tensor, op, cur_buft, cur_dev)) {
            return cur_buft;
        }
    }

    return nullptr;
}

// checks if the weight tensor can be used with the specified buffer type and device
static bool weight_buft_supported(const llama_hparams & hparams, ggml_tensor * w, ggml_op op, ggml_backend_buffer_type_t buft, ggml_backend_dev_t dev) {
   ...

    w->buffer = ggml_backend_buft_alloc_buffer(buft, 0);
    bool op_supported = ggml_backend_dev_supports_op(dev, op_tensor);
    ...
    return op_supported;
}
```
```c++
bool ggml_backend_dev_supports_op(ggml_backend_dev_t device, const struct ggml_tensor * op) {
    GGML_ASSERT(device);
    return device->iface.supports_op(device, op);
}
```
```c++
static bool ggml_backend_cpu_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];

    if (op->op == GGML_OP_NONE || op->op == GGML_OP_RESHAPE || op->op == GGML_OP_VIEW || op->op == GGML_OP_PERMUTE || op->op == GGML_OP_TRANSPOSE) {
        return true;
    }

    // check extra buffer types
    // note: only the first sources are checked for extra buffer types to reduce overhead, increase if necessary
    for (int i = 0; i < 4; i++) {
        if (op->src[i] && op->src[i]->buffer && ggml_backend_cpu_is_extra_buffer_type(op->src[i]->buffer->buft)) {
            auto * buf_extra = (ggml::cpu::extra_buffer_type *) op->src[i]->buffer->buft->context;
            return buf_extra->supports_op(dev, op);
        }
    }
```
This will call into `ggml/src/ggml-cpu/repack.cpp`:
```c++
namespace ggml::cpu::repack {

class extra_buffer_type : ggml::cpu::extra_buffer_type {

    bool supports_op(ggml_backend_dev_t, const struct ggml_tensor * op) override {
        if (    op->op == GGML_OP_MUL_MAT &&
                op->src[0]->buffer &&
                (ggml_n_dims(op->src[0]) == 2) &&
                op->src[0]->buffer->buft == ggml_backend_cpu_repack_buffer_type() &&
                ggml_repack_get_optimal_repack_type(op->src[0])
                ) {
            if (op->src[1]->buffer && !ggml_backend_buft_is_host(op->src[1]->buffer->buft)) {
                return false;
            }
            if (op->src[1]->type == GGML_TYPE_F32) {
                return true;
            }
            //if (op->src[1]->type == GGML_TYPE_Q8_0) {
            //    return true;
            //}
            // may be possible if Q8_0 packed...
        } else if (op->op == GGML_OP_MUL_MAT_ID
                && op->src[0]->buffer
                && (ggml_n_dims(op->src[0]) == 3)
                && op->src[0]->buffer->buft == ggml_backend_cpu_repack_buffer_type()
                && ggml_repack_get_optimal_repack_type(op->src[0])
                ) {
            if (op->src[1]->buffer && !ggml_backend_buft_is_host(op->src[1]->buffer->buft)) {
                return false;
            }
            if (op->src[1]->type == GGML_TYPE_F32) {
                return true;
            }
            //if (op->src[1]->type == GGML_TYPE_Q8_0) {
            //    return true;
            //}
        }
        return false;
    }

