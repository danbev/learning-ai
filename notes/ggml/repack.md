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

### Usage in ggml
In `ggml/CMakeLists.txt` we have the following option:
```console
option(GGML_CPU_REPACK       "ggml: use runtime weight conversion of Q4_0 to Q4_X_X" ON)
```
So this is enabled by default.

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
And it is also used in:
```c++
static ggml_backend_feature * ggml_backend_cpu_get_features(ggml_backend_reg_t reg) {
    static std::vector<ggml_backend_feature> features = []() {
        ggml_cpu_init();

        ...
    #ifdef GGML_USE_CPU_REPACK
        features.push_back({ "REPACK", "1" });
    #endif
```

But returning to the `ggml_backend_cpu_get_extra_buffer_types`, this is my first
time coming across this extra buffer type concept.

### Extra buffer type
Extra buffer types is the mechanism that is used to implement weight repacking
, see [repack.md](../ggml/repack.md), in the CPU backend. Basically, some weights
can be stored in a different buffer type, which repacks them in a way that is
more performant for the hardware it is running on.

So, in `ggml/src/ggml-cpu/ggml-cpu.cpp` we have:
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
    }(); // IIFE - Immediately Invoked Function Expression

    return bufts;
}
```
So in our case this this will call `ggml_backend_cpu_repack_buffer_type`:
```c++
ggml_backend_buffer_type_t ggml_backend_cpu_repack_buffer_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_cpu_buffer_type_repack = {
        /* .iface    = */ {
                           /* .get_name         = */ ggml_backend_cpu_repack_buffer_type_get_name,
                           /* .alloc_buffer     = */ ggml_backend_cpu_repack_buffer_type_alloc_buffer,
                           /* .get_alignment    = */ ggml_backend_cpu_repack_buffer_type_get_alignment,
                           /* .get_max_size     = */ nullptr,  // defaults to SIZE_MAX
                           /* .get_alloc_size   = */ nullptr,  // defaults to ggml_nbytes
                           /* .is_host          = */ nullptr,
                           },
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ new ggml::cpu::repack::extra_buffer_type(),
    };

    return &ggml_backend_cpu_buffer_type_repack;
}
```
Notice that the buffer type context will be set to a new instance of
`ggml::cpu::repack::extra_buffer_type`:
```
class extra_buffer_type : ggml::cpu::extra_buffer_type {

}
```
The declaration for `extra_buffer_type` can be found in `ggml/src/ggml-cpu/traits.h`:
```c++
class extra_buffer_type {
  public:
    virtual ~extra_buffer_type();
    virtual bool            supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) = 0;
    virtual tensor_traits * get_tensor_traits(const struct ggml_tensor * op)                   = 0;
};
```
So repack is a specific backend cpu buffer type with a context that implements
the `extra_buffer_type` interface.

This above function is called by:
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
```
Now, `extra_bufts` is a static vector, and it is initalized using an
immediately-invoked lambda expression (IIFE) (notice the ()). This will call
`ggml_backend_cpu_get_extra_buffer_types` where depending on the macros that
have been defined at compile time it will add the corresponding buffer types.
Then a `nullptr` is added to the end of the vector to enable iterating over
the vector until a `nullptr` is found.

<a name="loading"></a>
To get an overview of the process of loading tensors in `llama::model::load_tensors`
the steps are roughly:
* Creates the tensors
* Initializes the tensor backend buffers (ggml_backend_cpu_repack_buffer_init_tensor)
* Loads the tensors data (ggml_backend_cpu_repack_buffer_set_tensor)

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
Which in turn calls `make_cpu_buft_list`, and notice that this is passing
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
a string name, and in this case "ggml_backend_dev_get_extra_bufts" is passed in
as the name, and returns a void pointer:
```c++
void * ggml_backend_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_ASSERT(reg);
    if (!reg->iface.get_proc_address) {
        return NULL;
    }
    return reg->iface.get_proc_address(reg, name);
}
```
So if the interface of this backend registry has a `get_proc_address` function
then it will be called.

We can see that the passed in name is checked against a number of strings and
that they all set and return function pointers if there is a match:
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
The function returns a pointer to the first entry (and that is the reasons for
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
So using emplace back we add a std::pair of the cpu device and the buffer type.

So the `buft_list` will now contain the repack buffer type. But when will it
be used?  

When a tensor is created in `llama_model::load_tensors` for example, for
GEMMA3 we have:
```c++

            case LLM_ARCH_GEMMA3:
            ...
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
```

So lets unpack this a bit. The passed in tensor is the operation that to be
checked if it is supported. Currenty the only supported operations are
`GGML_OP_MUL_MAT`, and `GGML_OP_MUL_MAT_ID` and the number of dimensions of the
source tensor must be 2 or 3 respectively, and the buffer type of the source
tensor must be the repack.
Also, `ggml_repack_get_optimal_repack_type` must not return null for the source
tensor buffer.
For example:
```console
(gdb) p op->op
$27 = GGML_OP_MUL_MAT

(gdb) p op->src[0]->buffer
$28 = (ggml_backend_buffer *) 0x555555960af0

(gdb) p ggml_n_dims(op->src[0])
$29 = 2

(gdb) p op->src[0]->buffer->buft
$30 = (ggml_backend_buffer_type_t) 0x7ffff777b200 <ggml_backend_cpu_repack_buffer_type()::ggml_backend_cpu_buffer_type_repack>
(gdb) p op->src[0]->buffer->buft == ggml_backend_cpu_repack_buffer_type()
$31 = true

(gdb) p ggml_repack_get_optimal_repack_type(op->src[0])
$32 = (const ggml::cpu::tensor_traits *) 0x7ffff777b1c8 <ggml_repack_get_optimal_repack_type(ggml_tensor const*)::q4_0_8x8_q8_0>
```
This would return true.

```c++
static const ggml::cpu::tensor_traits * ggml_repack_get_optimal_repack_type(const struct ggml_tensor * cur) {
    // instance for Q4
    static const ggml::cpu::repack::tensor_traits<block_q4_0, 4, 4, GGML_TYPE_Q8_0> q4_0_4x4_q8_0;
    static const ggml::cpu::repack::tensor_traits<block_q4_0, 8, 4, GGML_TYPE_Q8_0> q4_0_4x8_q8_0;
    static const ggml::cpu::repack::tensor_traits<block_q4_0, 8, 8, GGML_TYPE_Q8_0> q4_0_8x8_q8_0;
    static const ggml::cpu::repack::tensor_traits<block_q4_K, 8, 8, GGML_TYPE_Q8_K> q4_K_8x8_q8_K;

    // instance for Q2
    static const ggml::cpu::repack::tensor_traits<block_q2_K, 8, 8, GGML_TYPE_Q8_K> q2_K_8x8_q8_K;

    // instance for IQ4
    static const ggml::cpu::repack::tensor_traits<block_iq4_nl, 4, 4, GGML_TYPE_Q8_0> iq4_nl_4x4_q8_0;
    static const ggml::cpu::repack::tensor_traits<block_iq4_nl, 8, 8, GGML_TYPE_Q8_0> iq4_nl_8x8_q8_0;
```
The traits are different repacking configurations, the blocktype, the block
dimensions, and the target quantization type.

Just recall that the repacking is about taking a quantized tensor and seeing if
it can be more optimally repacked for the operations specific to the hardware
in use.
```c++
    if (cur->type == GGML_TYPE_Q4_0) {
        if (ggml_cpu_has_avx2() || (ggml_cpu_has_sve() && ggml_cpu_has_matmul_int8() && ggml_cpu_get_sve_cnt() == QK8_0)) {
            if (cur->ne[1] % 8 == 0) {
                return &q4_0_8x8_q8_0;
            }
        }
        if (ggml_cpu_has_neon() && ggml_cpu_has_matmul_int8()) {
            if (cur->ne[1] % 4 == 0) {
                return &q4_0_4x8_q8_0;
            }
        }
        if (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
            if (cur->ne[1] % 4 == 0) {
                return &q4_0_4x4_q8_0;
            }
        }
    } else if (cur->type == GGML_TYPE_Q4_K) {
        if (ggml_cpu_has_avx2()) {
            if (cur->ne[1] % 8 == 0) {
                return &q4_K_8x8_q8_K;
            }
        }
    } else if (cur->type == GGML_TYPE_Q2_K) {
        if (ggml_cpu_has_avx512()) {
            if (cur->ne[1] % 8 == 0) {
                return &q2_K_8x8_q8_K;
            }
        }
    } else if (cur->type == GGML_TYPE_IQ4_NL) {
        if (ggml_cpu_has_avx2()) {
            if (cur->ne[1] % 8 == 0) {
                return &iq4_nl_8x8_q8_0;
            }
        }
        if (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
            if (cur->ne[1] % 4 == 0) {
                return &iq4_nl_4x4_q8_0;
            }
        }
    }

    return nullptr;
}
```
So that was the creation of the tensors. Next step is to initialize the tensor.

Notice that the traits are stored in the `extra` field of the source tensor which
is the first time I've seen this be used in practice and something that I missed
when going through repack above, but this is set in:
```c++
static enum ggml_status ggml_backend_cpu_repack_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    tensor->extra = (void *) const_cast<ggml::cpu::tensor_traits *>(ggml_repack_get_optimal_repack_type(tensor));

    GGML_UNUSED(buffer);
    return GGML_STATUS_SUCCESS;
}
```
This is called when the tensors are allocated in `llama_model::load_tensors`:
```c++
bool llama_model::load_tensors(llama_model_loader & ml) {
    ...
    // create tensors for the weights
    {

        const auto tn = LLM_TN(arch);
        switch (arch) {
            case LLM_ARCH_LLAMA:
            case LLM_ARCH_REFACT:
            case LLM_ARCH_MINICPM:
            case LLM_ARCH_GRANITE:
            case LLM_ARCH_GRANITE_MOE:
            ...
        }
    }
    ml.done_getting_tensors();
    ...
    for (auto & it : ctx_map) {
        ggml_backend_buffer_type_t buft = it.first;
        ggml_context * ctx              = it.second;
        ...
        if (ml.use_mmap && use_mmap_buffer && buffer_from_host_ptr_supported && is_default_buft) {
            ...
        }
        else {
            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
            ...
        }
}
```
```c++
ggml_backend_buffer_t ggml_backend_alloc_ctx_tensors_from_buft(struct ggml_context * ctx, ggml_backend_buffer_type_t buft) {
    ...
        if (!alloc_tensor_range(ctx, first, NULL, buft, cur_buf_size, &buffers, &n_buffers)) {
            ....
        }
```
```c++
static bool alloc_tensor_range(struct ggml_context * ctx,
        struct ggml_tensor * first, struct ggml_tensor * last,
        ggml_backend_buffer_type_t buft, size_t size,
        ggml_backend_buffer_t ** buffers, size_t * n_buffers) {
        ...
    for (struct ggml_tensor * t = first; t != last; t = ggml_get_next_tensor(ctx, t)) {
        enum ggml_status status = GGML_STATUS_SUCCESS;
        if (t->data == NULL) {
            if (t->view_src == NULL) {
                status = ggml_tallocr_alloc(&tallocr, t);
```

```c++
enum ggml_status ggml_tallocr_alloc(struct ggml_tallocr * talloc, struct ggml_tensor * tensor) {
    size_t size = ggml_backend_buffer_get_alloc_size(talloc->buffer, tensor);
    size = GGML_PAD(size, talloc->alignment);

    void * addr = (char *)ggml_backend_buffer_get_base(talloc->buffer) + talloc->offset;
    talloc->offset += size;

    return ggml_backend_tensor_alloc(talloc->buffer, tensor, addr);
}

enum ggml_status ggml_backend_tensor_alloc(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, void * addr) {
    ...
    tensor->buffer = buffer;
    tensor->data = addr;
    return ggml_backend_buffer_init_tensor(buffer, tensor);
}

enum ggml_status ggml_backend_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    GGML_ASSERT(buffer);
    // init_tensor is optional
    if (buffer->iface.init_tensor) {
        return buffer->iface.init_tensor(buffer, tensor);
    }
    return GGML_STATUS_SUCCESS;
}
```
<a name="init"></a>
And `init_tensor` is the function `ggml_backend_cpu_repack_buffer_init_tensor`: 
```c++
static enum ggml_status ggml_backend_cpu_repack_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    tensor->extra = (void *) const_cast<ggml::cpu::tensor_traits *>(ggml_repack_get_optimal_repack_type(tensor));

    GGML_UNUSED(buffer);
    return GGML_STATUS_SUCCESS;
}
```
So this is how the tensor->extra field is set.
And `init_tensor` is set in:
```c++
static ggml_backend_buffer_t ggml_backend_cpu_repack_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);

    if (buffer == nullptr) {
        return nullptr;
    }

    buffer->buft              = buft;
    buffer->iface.init_tensor = ggml_backend_cpu_repack_buffer_init_tensor;
    buffer->iface.set_tensor  = ggml_backend_cpu_repack_buffer_set_tensor;
    buffer->iface.get_tensor  = nullptr;
    buffer->iface.cpy_tensor  = nullptr;
    return buffer;
}
```

<a name="repacking"></a>
This is where the repackaging actually happens:
```c++
static void ggml_backend_cpu_repack_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor,
                                                       const void * data, size_t offset, size_t size) {
    auto tensor_traits = (ggml::cpu::repack::tensor_traits_base *) tensor->extra;
    auto OK            = tensor_traits->repack(tensor, data, size);

    GGML_ASSERT(OK == 0);
    GGML_UNUSED(buffer);
}
```
So it is at tensor data load time that the repacking takes place. For example:
```console
(gdb) p *tensor
$40 = {type = GGML_TYPE_Q4_0, buffer = 0x555555967f50, ne = {640, 1024, 1, 1}, nb = {18, 360, 368640, 368640}, op = GGML_OP_NONE, 
op_params = {0 <repeats 16 times>}, flags = 0, src = {0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0}, view_src = 0x0, 
view_offs = 0, data = 0x7fffdda35040, name = "blk.0.attn_q.weight", '\000' <repeats 44 times>, 
extra = 0x7ffff777b1c8 <ggml_repack_get_optimal_repack_type(ggml_tensor const*)::q4_0_8x8_q8_0>, 
padding = "\000\000\000\000\000\000\000"}
```
```c++
    int repack(struct ggml_tensor * t, const void * data, size_t data_size) override {
        GGML_LOG_DEBUG("%s: repack tensor %s with %s_%dx%d\n", __func__, t->name, ggml_type_name(t->type),
                       (int) NB_COLS, (int) INTER_SIZE);
        return ggml::cpu::repack::repack<BLOC_TYPE, INTER_SIZE, NB_COLS>(t, data, data_size);
    }

template <> int repack<block_q4_0, 8, 8>(struct ggml_tensor * t, const void * data, size_t data_size) {
    return repack_q4_0_to_q4_0_8_bl(t, 8, data, data_size);
}
```
And this is where the input data the tensor data should be set to is repackaged
into a format that is more optimal for the hardware. So we have our original
data in Q4_0 format and we want to repack it block_q4_0x8 format for more
optimized SIMD operations. This will become important later when we look at the
matrix multiplication operation and see that it operates on 8 rows at a time.
```c++
static int repack_q4_0_to_q4_0_8_bl(struct ggml_tensor * t, int interleave_block, const void * GGML_RESTRICT data, size_t data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(interleave_block == 8);
    constexpr int nrows_interleaved = 8;

    block_q4_0x8 * dst = (block_q4_0x8*)t->data;
    const block_q4_0 * src = (const block_q4_0*) data;
    block_q4_0 dst_tmp[8];
    int nrow = ggml_nrows(t);
    int nblocks = t->ne[0] / QK4_0;

    GGML_ASSERT(data_size == nrow * nblocks * sizeof(block_q4_0));

    if (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        return -1;
    }

    for (int b = 0; b < nrow; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; x++) {
            for (int i  = 0; i < nrows_interleaved; i++ ) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_q4_0x8(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    return 0;

    GGML_UNUSED(data_size);
}
```

So to recap at startup when loading model tensors, we first detemine if the
backend can handle the tensor, and if so when the backend buffer initializes
the tensor it will set the tensor extra field to the repack traits. And when
the tensor is populated with data the set tensor function will be called which
will perform the actualy repacking.

So that was the loading of tensors, next is the usage in matrix multiplication
operations where the repacked format is used to speed up the operations.

To see this in action we need a model that has quantized weights, for example
I'll use `gemma-3-270m-it-qat-q4_0-unquantized-Q4_0.gguf` for this.

Then we can set a break point:
```console
(gdb) br repack.cpp:1604
```
The call stack will look something like this, `llama_decode` in `ggml-context.cpp`
will call `process_ubatch`:
```c++
llm_graph_result * llama_context::process_ubatch(const llama_ubatch & ubatch, llm_graph_type gtype, llama_memory_context_i * mctx, ggml_status & ret) {
    ...

    const auto status = graph_compute(res->get_gf(), ubatch.n_tokens > 1);
    ...
}
```
```c++
ggml_status llama_context::graph_compute(
            ggml_cgraph * gf,
                   bool   batched) {
    ...
    auto status = ggml_backend_sched_graph_compute_async(sched.get(), gf);
    ...
```
```c++
enum ggml_status ggml_backend_sched_graph_compute_async(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    ...
    return ggml_backend_sched_compute_splits(sched);
}
```
```c++
static enum ggml_status ggml_backend_sched_compute_splits(ggml_backend_sched_t sched) {
    ...
        if (!sched->callback_eval) {
            enum ggml_status ec = ggml_backend_graph_compute_async(split_backend, &split->graph);
            if (ec != GGML_STATUS_SUCCESS) {
                return ec;
            }
```
```c++
enum ggml_status ggml_backend_graph_compute_async(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    return backend->iface.graph_compute(backend, cgraph);
}
```
```console
(gdb) p backend.iface.get_name(backend)
$3 = 0x7ffff77433f7 "CPU"
```
This will land us in `ggml-cpu.cpp`:
```c++
static enum ggml_status ggml_backend_cpu_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    ...
    return ggml_graph_compute(cgraph, &cplan);
}
```
Which will call into `ggml-cpu.c`:
```c++
enum ggml_status ggml_graph_compute(struct ggml_cgraph * cgraph, struct ggml_cplan * cplan) {
    ...
#ifdef GGML_USE_OPENMP
    if (n_threads > 1) {
        #pragma omp parallel num_threads(n_threads)
        {
            #pragma omp single
            {
                // update the number of threads from the actual number of threads that we got from OpenMP
                n_threads = omp_get_num_threads();
                atomic_store_explicit(&threadpool->n_threads_cur, n_threads, memory_order_relaxed);
            }

            ggml_graph_compute_thread(&threadpool->workers[omp_get_thread_num()]);
        }
    } else {
        atomic_store_explicit(&threadpool->n_threads_cur, 1, memory_order_relaxed);
        ggml_graph_compute_thread(&threadpool->workers[0]);
    }
#else
    if (n_threads > threadpool->n_threads_max) {
        GGML_LOG_WARN("cplan requested more threads (%d) than available (%d)\n", n_threads, threadpool->n_threads_max);
        n_threads = threadpool->n_threads_max;
    }

    // Kick all threads to start the new graph
    ggml_graph_compute_kickoff(threadpool, n_threads);

    // This is a work thread too
    ggml_graph_compute_thread(&threadpool->workers[0]);
#endif
   ...
}
```
And the above will start new threads that will perform the graph computation and
the code for that is in:
```c++
static thread_ret_t ggml_graph_compute_thread(void * data) {
    struct ggml_compute_state * state = (struct ggml_compute_state *) data;
    struct ggml_threadpool    * tp    = state->threadpool;

    for (int node_n = 0; node_n < cgraph->n_nodes && atomic_load_explicit(&tp->abort, memory_order_relaxed) != node_n; node_n++) {
        struct ggml_tensor * node = cgraph->nodes[node_n];

        ggml_compute_forward(&params, node);
        ...
```
And this brings us to:
```c++
static void ggml_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * tensor) {
    GGML_ASSERT(params);

    if (tensor->op == GGML_OP_NONE || ggml_is_empty(tensor)) {
        return;
    }

    // extra_buffer op?
    if (ggml_cpu_extra_compute_forward(params, tensor)) {
        return;
    }
```

```c++
bool ggml_cpu_extra_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * op) {
    for (auto extra : ggml_backend_cpu_get_extra_buffer_types()) {
        if (extra && extra->context) {
            auto buf_extra     = (ggml::cpu::extra_buffer_type *) extra->context;
            auto tensor_traits = buf_extra->get_tensor_traits(op);
            if (tensor_traits && tensor_traits->compute_forward(params, op)) {
                return true;
            }
        }
    }
    return false;
}
```
Now, the `tensor_traits` will be retrieved from the `src[0]` extra field:
```c++
    ggml::cpu::tensor_traits * get_tensor_traits(const struct ggml_tensor * op) override {
        if (op->op == GGML_OP_MUL_MAT || op->op == GGML_OP_MUL_MAT_ID) {
            if (op->src[0]->buffer && op->src[0]->buffer->buft == ggml_backend_cpu_repack_buffer_type()) {
                return (ggml::cpu::tensor_traits *) op->src[0]->extra;
            }
        }
        return nullptr;
    }
```
And if there is such a traits instance, then its `compute_forward` function will
be called:
```c++
    bool compute_forward(struct ggml_compute_params * params, struct ggml_tensor * op) override {
        switch (op->op) {
            case GGML_OP_MUL_MAT:
                forward_mul_mat(params, op);
                return true;
            case GGML_OP_MUL_MAT_ID:
                forward_mul_mat_id(params, op);
                return true;
            default:
                // GGML_ABORT("fatal error");
                break;
        }
        return false;
    }
```
Lets take a look at an example of tensor for this operation:
```console
(gdb) p *op
$2 = {type = GGML_TYPE_F32, buffer = 0x55555596a280, ne = {1024, 9, 1, 1},
nb = {4, 4096, 36864, 36864}, op = GGML_OP_MUL_MAT, op_params = {0 <repeats 16 times>},
flags = 0, src = {0x55555836b6e0, 0x555555d1cd00, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0}, 
view_src = 0x0, view_offs = 0, data = 0x7fffbb217840, name = "Qcur-0", '\000' <repeats 57 times>, extra = 0x0, 
padding = "\000\000\000\000\000\000\000"}

(gdb) p op->src[0]->name
$3 = "blk.0.attn_q.weight", '\000' <repeats 44 times>
(gdb) p op->src[0]->type
$4 = GGML_TYPE_Q4_0
(gdb) p op->src[0]->ne
$5 = {640, 1024, 1, 1}

(gdb) p op->src[1]->name
$6 = "attn_norm-0", '\000' <repeats 52 times>
(gdb) p op->src[1]->type
$7 = GGML_TYPE_F32
(gdb) p op->src[1]->ne
$9 = {640, 9, 1, 1}
```
Notice how `src[0]` is of type Q4_0 and has dimensions 640x1024, and `src[1]` is
of type F32.
And params are:
```console
(gdb) p *params
$11 = {ith = 1, nth = 4, wsize = 19840, wdata = 0x55555859b9b0, threadpool = 0x555555968240}
```

And further down we have the `forward_mul_mat` function which is the operation
that we are currently stepping through:
```
    void forward_mul_mat(ggml_compute_params * params, ggml_tensor * op) {
        const ggml_tensor * src0 = op->src[0];
        const ggml_tensor * src1 = op->src[1];
        ggml_tensor *       dst  = op;

```
To avoid switching to an different thread when stepping through the code in gdb,
we can enable scheduler locking in gdb:
```console
(gdb) set scheduler-locking on
```
Again, we can inspect the tensors:
```c++
(gdb) p *src0
$10 = {type = GGML_TYPE_Q4_0, buffer = 0x555555dab150,
ne = {640, 1024, 1, 1}, nb = {18, 360, 368640, 368640}, op = GGML_OP_NONE,
op_params = {0 <repeats 16 times>}, flags = 0, src = {0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0}, view_src = 0x0,
view_offs = 0, data = 0x7fffb8a35040, name = "blk.0.attn_q.weight", '\000' <repeats 44 times>,
extra = 0x7ffff777b1a8 <ggml_repack_get_optimal_repack_type(ggml_tensor const*)::q4_0_8x8_q8_0>,
padding = "\000\000\000\000\000\000\000"}
```

```console
(gdb) p *src1
$11 = {type = GGML_TYPE_F32, buffer = 0x555555dac130,
ne = {640, 9, 1, 1}, nb = {4, 2560, 23040, 23040}, op = GGML_OP_MUL,
op_params = {0 <repeats 16 times>}, flags = 0, src = {0x555556156750, 0x555558764dc0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0},
view_src = 0x0, view_offs = 0, data = 0x7fff834db840, name = "attn_norm-0", '\000' <repeats 52 times>, extra = 0x0,
padding = "\000\000\000\000\000\000\000"}
```

```console
(gdb) p *dst
$12 = {type = GGML_TYPE_F32, buffer = 0x555555dac130, ne = {1024, 9, 1, 1}, nb = {4, 4096, 36864, 36864}, op = GGML_OP_MUL_MAT, 
  op_params = {0 <repeats 16 times>}, flags = 0, src = {0x55555877d3a0, 0x5555561568c0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0}, 
  view_src = 0x0, view_offs = 0, data = 0x7fff8361b840, name = "Qcur-0", '\000' <repeats 57 times>, extra = 0x0, 
  padding = "\000\000\000\000\000\000\000"}
```

```c++
        const int ith = params->ith;  // Current thread id (0, 1, 2, ..., nth-1).
        const int nth = params->nth;  // Total number of threads.
```
```console
(gdb) p ith
$19 = 1
(gdb) p nth
$18 = 4
```

Next we have the work data.
```c++
        char *       wdata = static_cast<char *>(params->wdata);
        const size_t nbw1  = ggml_row_size(PARAM_TYPE, ne10);
```

```console
(gdb) p nbw1
$16 = 680
```

Next we get the function from the trait which will be used to quantize `src[1]`.
So `src[0]` is already quantized, but `src[1]` is not and the matrix multiplication
is performed on the quantized types so we need this conversion.
```c++
        const ggml_from_float_t from_float = ggml_get_type_traits_cpu(PARAM_TYPE)->from_float;
```

```console
(gdb) p from_float
$17 = (const ggml_from_float_t) 0x7ffff76f2775 <quantize_row_q8_0>
```
```c++
        int64_t i11_processed = 0;
        // Each thread will process every nth group of 4 rows.
        for (int64_t i11 = ith * 4; i11 < ne11 - ne11 % 4; i11 += nth * 4) {
            ggml_quantize_mat_t<INTER_SIZE, PARAM_TYPE>(
                (float *) ((char *) src1->data + i11 * nb11),  // x
                (void *) (wdata + i11 * nbw1),                 // vy
                4,                                             // n_rows (4)
                ne10);                                         // (640)
        }
```
Notice that `src1` is being passed in as the first argument x:
```console
(gdb) p src1->name
$39 = "attn_norm-0", '\000' <repeats 52 times>
(gdb) p src1->type
$40 = GGML_TYPE_F32
```
So this is actually going to quantize src1, and we process all the groups of 4
in `src1`, then we have the leftovers which we just use `from_float` on because
they can't fit into a quantized block. Then we wait for all threads to reach the
barrier point.
The quantized output will be placed in wdata.

```c++
template <> void ggml_quantize_mat_t<8, GGML_TYPE_Q8_0>(
    const float * GGML_RESTRICT x,
    void * GGML_RESTRICT vy,
    int64_t nrow,
    int64_t n_per_row) {
    assert(nrow == 4);
    UNUSED(nrow);
    ggml_quantize_mat_q8_0_4x8(x, vy, n_per_row);
}
```
Now, this will call into `ggml/src/ggml-cpu/arch/x86/repack.cpp`:
```c++
void ggml_quantize_mat_q8_0_4x8(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(QK8_0 == 32);
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;

    block_q8_0x4 * GGML_RESTRICT y = (block_q8_0x4 *) vy;
```
```console
(gdb) p k
$36 = 640
(gdb) p nb
$37 = 20
```
I'm going to got through this function and comment on what is happening but
basically this is quantizing like we have seen before but doing it in an optimal
way for the hardware in question. The quantization that I've looked at previously
was used when quantizing models for storage where as this is quantization for
runtime usage.
```c++
void ggml_quantize_mat_q8_0_4x8(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(QK8_0 == 32);
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;

    block_q8_0x4 * GGML_RESTRICT y = (block_q8_0x4 *) vy;

#if defined(__AVX2__) || defined(__AVX__)
    float id[4];
    __m256 srcv[4][4];
    __m256 idvec[4];

    for (int i = 0; i < nb; i++) {
        for (int row_iter = 0; row_iter < 4; row_iter++) {
            // Load elements into 4 AVX vectors
            __m256 v0 = _mm256_loadu_ps( x + row_iter * k + i * 32 );
            __m256 v1 = _mm256_loadu_ps( x + row_iter * k + i * 32 + 8 );
            __m256 v2 = _mm256_loadu_ps( x + row_iter * k + i * 32 + 16 );
            __m256 v3 = _mm256_loadu_ps( x + row_iter * k + i * 32 + 24 );

            // Compute max(abs(e)) for the block
            const __m256 signBit = _mm256_set1_ps( -0.0f );
            __m256 maxAbs = _mm256_andnot_ps( signBit, v0 );
            maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v1 ) );
            maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v2 ) );
            maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v3 ) );

            __m128 max4 = _mm_max_ps( _mm256_extractf128_ps( maxAbs, 1 ), _mm256_castps256_ps128( maxAbs ) );
            max4 = _mm_max_ps( max4, _mm_movehl_ps( max4, max4 ) );
            max4 = _mm_max_ss( max4, _mm_movehdup_ps( max4 ) );
            const float maxScalar = _mm_cvtss_f32( max4 );

            // Divided by 127.f to mirror results in quantize_row_q8_0
            const float d = maxScalar  / 127.f;
            id[row_iter] = ( maxScalar != 0.0f ) ? 127.f / maxScalar : 0.0f; //d ? 1.0f / d : 0.0f;

            // Store the scale for the individual block
            y[i].d[row_iter] = GGML_CPU_FP32_TO_FP16(d);

            // Store the values in blocks of eight values - Aim is to use these later for block interleaving
            srcv[row_iter][0] = v0;
            srcv[row_iter][1] = v1;
            srcv[row_iter][2] = v2;
            srcv[row_iter][3] = v3;
            idvec[row_iter] = _mm256_set1_ps(id[row_iter]);
        }

        // The loop iterates four times - The aim is to get 4 corresponding chunks of eight bytes from the original weight blocks that are interleaved
        for (int j = 0; j < 4; j++) {
            // Apply the multiplier
            __m256 v0 = _mm256_mul_ps(srcv[0][j], idvec[0]);
            __m256 v1 = _mm256_mul_ps(srcv[1][j], idvec[1]);
            __m256 v2 = _mm256_mul_ps(srcv[2][j], idvec[2]);
            __m256 v3 = _mm256_mul_ps(srcv[3][j], idvec[3]);

            // Round to nearest integer
            v0 = _mm256_round_ps( v0, _MM_ROUND_NEAREST );
            v1 = _mm256_round_ps( v1, _MM_ROUND_NEAREST );
            v2 = _mm256_round_ps( v2, _MM_ROUND_NEAREST );
            v3 = _mm256_round_ps( v3, _MM_ROUND_NEAREST );

            // Convert floats to integers
            __m256i i0 = _mm256_cvtps_epi32( v0 );
            __m256i i1 = _mm256_cvtps_epi32( v1 );
            __m256i i2 = _mm256_cvtps_epi32( v2 );
            __m256i i3 = _mm256_cvtps_epi32( v3 );

#if defined(__AVX2__)
            // Convert int32 to int16
            i0 = _mm256_packs_epi32( i0, i1 );
            i2 = _mm256_packs_epi32( i2, i3 );
            // Convert int16 to int8
            i0 = _mm256_packs_epi16( i0, i2 );

            //  Permute and store the quantized weights in the required order after the pack instruction
            const __m256i perm = _mm256_setr_epi32( 0, 4, 1, 5, 2, 6, 3, 7 );
            i0 = _mm256_permutevar8x32_epi32( i0, perm );

            _mm256_storeu_si256((__m256i *)(y[i].qs + 32 * j), i0);
#else
            // Since we don't have in AVX some necessary functions,
            // we split the registers in half and call AVX2 analogs from SSE
            __m128i ni0 = _mm256_castsi256_si128( i0 );
            __m128i ni1 = _mm256_extractf128_si256( i0, 1);
            __m128i ni2 = _mm256_castsi256_si128( i1 );
            __m128i ni3 = _mm256_extractf128_si256( i1, 1);
            __m128i ni4 = _mm256_castsi256_si128( i2 );
            __m128i ni5 = _mm256_extractf128_si256( i2, 1);
            __m128i ni6 = _mm256_castsi256_si128( i3 );
            __m128i ni7 = _mm256_extractf128_si256( i3, 1);

            // Convert int32 to int16
            ni0 = _mm_packs_epi32( ni0, ni1 );
            ni2 = _mm_packs_epi32( ni2, ni3 );
            ni4 = _mm_packs_epi32( ni4, ni5 );
            ni6 = _mm_packs_epi32( ni6, ni7 );
            // Convert int16 to int8
            ni0 = _mm_packs_epi16( ni0, ni2 );
            ni4 = _mm_packs_epi16( ni4, ni6 );
            _mm_storeu_si128((__m128i *)(y[i].qs + 32 * j), ni0);
            _mm_storeu_si128((__m128i *)(y[i].qs + 32 * j + 16), ni4);
#endif
        }
    }

#else
    UNUSED(nb);
    UNUSED(y);
    ggml_quantize_mat_q8_0_4x8_generic(x, vy, k);
#endif
}
```
After this the leftover rows are processed:
```c++
        i11_processed = ne11 - ne11 % 4;
        for (int64_t i11 = i11_processed + ith; i11 < ne11; i11 += nth) {
            from_float((float *) ((char *) src1->data + i11 * nb11), (void *) (wdata + i11 * nbw1), ne10);
        }
```
And the we wait for all threads to reach the barrier:
```c++
        ggml_barrier(params->threadpool);
```
```c++
        const void * src1_wdata      = params->wdata;
        const size_t src1_col_stride = ggml_row_size(PARAM_TYPE, ne10);
        int64_t      src0_start      = (ith * ne01) / nth;
        int64_t      src0_end        = ((ith + 1) * ne01) / nth;
        src0_start = (src0_start % NB_COLS) ? src0_start + NB_COLS - (src0_start % NB_COLS) : src0_start;
        src0_end   = (src0_end   % NB_COLS) ? src0_end   + NB_COLS - (src0_end   % NB_COLS) : src0_end;
        if (src0_start >= src0_end) {
            return;
        }

        // If there are more than three rows in src1, use gemm; otherwise, use gemv.
        if (ne11 > 3) {
            gemm<BLOC_TYPE, INTER_SIZE, NB_COLS, PARAM_TYPE>(ne00,
                    (float *) ((char *) dst->data) + src0_start, ne01,
                    (const char *) src0->data + src0_start * nb01,
                    (const char *) src1_wdata,
                    ne11 - ne11 % 4,
                    src0_end - src0_start);
        }
```
And notice that we are passing in:
```console
(gdb) p ne00
$41 = 640
(gdb) p dst->name
$43 = "Qcur-0", '\000' <repeats 57 times>
(gdb) p dst->type
$46 = GGML_TYPE_F32

(gdb) p src0->name
$44 = "blk.0.attn_q.weight", '\000' <repeats 44 times>
(gdb) p src0->type
$45 = GGML_TYPE_Q4_0

(gdb) p src1_wdata                    <----
$49 = (const void *) 0x5555589ab630

(gdb) p ne11 - ne11 % 4
$50 = 8
(gdb) p src0_end - src0_start
$51 = 256
```


This will call:
```c++
template <> void gemm<block_q4_0, 8, 8, GGML_TYPE_Q8_0>(
    int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemm_q4_0_8x8_q8_0(n, s, bs, vx, vy, nr, nc);
}
```

And just to clarify the naming of variables, this comes from standard BLAS
naming conventions:
* n  = width of the matrices
* s  = the output matrix, think sum or result
* bs = byte stride of the destination matrix (row stride in bytes)
* vx = Matrix A (left operand), "vector x" in BLAS terminology
* vy = Matrix B (right operand), "vector y" in BLAS terminology
* nr = number of rows to process
* nc = number of columns to process

And this will land in `ggml-cpu/arch/x86/repack.cpp`:
```c++
void ggml_gemm_q4_0_8x8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
#if defined(__AVX2__) || defined(__AVX512F__)
    {
        // Lookup table to convert signed nibbles to signed bytes
        __m256i signextendlut = _mm256_castsi128_si256(_mm_set_epi8(-1, -2, -3, -4, -5, -6, -7, -8, 7, 6, 5, 4, 3, 2, 1, 0));
        signextendlut = _mm256_permute2f128_si256(signextendlut, signextendlut, 0);

        gemm_q4_b32_8x8_q8_0_lut_avx<block_q4_0x8>(n, s, bs, vx, vy, nr, nc, signextendlut);

        return;
    }
#endif // defined(__AVX2__) || defined(__AVX512F__)

    ggml_gemm_q4_0_8x8_q8_0_generic(n, s, bs, vx, vy, nr, nc);
}
```

```c++
// GEMM for 8x blocks of 32 4-bit quants with a single scale factor per block
static void gemm_q4_b32_8x8_q8_0_lut_avx(int n,
    float * GGML_RESTRICT s,
    size_t bs,
    const void * GGML_RESTRICT vx,
    const void * GGML_RESTRICT vy,
    int nr,
    int nc,
    __m256i signextendlut) {

    const int qk = QK8_0;
    const int nb = n / qk;

    const block_tx8    * b_ptr_start = (const block_tx8    *)vx;
    const block_q8_0x4 * a_ptr_start = (const block_q8_0x4 *)vy;

    int64_t b_nb = n / 32;
    int64_t y = 0;
    // Mask to mask out nibbles from packed bytes
    const __m256i m4b = _mm256_set1_epi8(0x0F);
    const __m128i loadMask = _mm_blend_epi32(_mm_setzero_si128(), _mm_set1_epi32(0xFFFFFFFF), 3);
    // Permute mask used for easier vector processing at later stages
    __m256i requiredOrder = _mm256_set_epi32(3, 2, 1, 0, 7, 6, 5, 4);
    int64_t xstart = 0;
    int anr = nr - nr%16; // Used to align nr with boundary of 16

    // Take group of four block_q8_0x4 structures at each pass of the loop and perform dot product operation

    for (; y < anr / 4; y += 4) {
        const block_q8_0x4 * a_ptrs[4];

        a_ptrs[0] = a_ptr_start + (y * nb);
        for (int i = 0; i < 3; ++i) {
            a_ptrs[i + 1] = a_ptrs[i] + nb;
        }

        // Take group of eight block_tx8 structures at each pass of the loop and perform dot product operation
        for (int64_t x = xstart; x < nc / 8; x++) {

            const block_tx8 * b_ptr = b_ptr_start + (x * b_nb);

            // Master FP accumulators
            __m256 acc_rows[16];
            for (int i = 0; i < 16; i++) {
                acc_rows[i] = _mm256_setzero_ps();
            }

            for (int64_t b = 0; b < nb; b++) {
                // Load the eight blocks of quantized values interleaved with each other in chunks of eight - B0,B1 ....B6,B7
                const __m256i rhs_raw_mat_0123_0 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs));
                const __m256i rhs_raw_mat_4567_0 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 32));
                const __m256i rhs_raw_mat_0123_1 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 64));
                const __m256i rhs_raw_mat_4567_1 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 96));
```
The above is loading the quantized values, in the qs array, and doing so 32 bytes
at a time (32 * 8 = 256 bits). Each block_tx8 structure has 128 bytes

```c++

                // Save the values in the following vectors in the formats B0B1B4B5, B2B3B6B7 for further processing and storing of values
                const __m256i rhs_raw_mat_0145_0 = _mm256_blend_epi32(rhs_raw_mat_0123_0, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_0, requiredOrder), 240);
                const __m256i rhs_raw_mat_2367_0 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_0, requiredOrder), rhs_raw_mat_4567_0, 240);
                const __m256i rhs_raw_mat_0145_1 = _mm256_blend_epi32(rhs_raw_mat_0123_1, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_1, requiredOrder), 240);
                const __m256i rhs_raw_mat_2367_1 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_1, requiredOrder), rhs_raw_mat_4567_1, 240);

                // 4-bit -> 8-bit - Sign is maintained
                const __m256i rhs_mat_0145_0 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(rhs_raw_mat_0145_0, m4b)); //B0(0-7) B1(0-7) B4(0-7) B5(0-7)
                const __m256i rhs_mat_2367_0 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(rhs_raw_mat_2367_0, m4b)); //B2(0-7) B3(0-7) B6(0-7) B7(0-7)

                const __m256i rhs_mat_0145_1 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(rhs_raw_mat_0145_1, m4b)); //B0(8-15) B1(8-15) B4(8-15) B5(8-15)
                const __m256i rhs_mat_2367_1 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(rhs_raw_mat_2367_1, m4b)); //B2(8-15) B3(8-15) B6(8-15) B7(8-15)

                const __m256i rhs_mat_0145_2 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_0, 4), m4b)); //B0(16-23) B1(16-23) B4(16-23) B5(16-23)
                const __m256i rhs_mat_2367_2 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_0, 4), m4b)); //B2(16-23) B3(16-23) B6(16-23) B7(16-23)

                const __m256i rhs_mat_0145_3 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_1, 4), m4b)); //B0(24-31) B1(24-31) B4(24-31) B5(24-31)
                const __m256i rhs_mat_2367_3 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_1, 4), m4b)); //B2(24-31) B3(24-31) B6(24-31) B7(24-31)

                // Shuffle pattern one - right side input
                const __m256i rhs_mat_0145_0_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_0, 136);  //B0(0-3) B1(0-3) B0(0-3) B1(0-3) B4(0-3) B5(0-3) B4(0-3) B5(0-3)
                const __m256i rhs_mat_2367_0_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_0, 136);  //B2(0-3) B3(0-3) B2(0-3) B3(0-3) B6(0-3) B7(0-3) B6(0-3) B7(0-3)

                const __m256i rhs_mat_0145_1_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_1, 136);  //B0(8-11) B1(8-11) B0(8-11) B1(8-11) B4(8-11) B5(8-11) B4(8-11) B5(8-11)
                const __m256i rhs_mat_2367_1_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_1, 136);  //B2(8-11) B3(8-11) B2(8-11) B3(8-11) B6(8-11) B7(8-11) B6(8-11) B7(8-11)

                const __m256i rhs_mat_0145_2_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_2, 136);  //B0(16-19) B1(16-19) B0(16-19) B1(16-19) B4(16-19) B5(16-19) B4(16-19) B5(16-19)
                const __m256i rhs_mat_2367_2_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_2, 136);  //B2(16-19) B3(16-19) B2(16-19) B3(16-19) B6(16-19) B7(16-19) B6(16-19) B7(16-19)

                const __m256i rhs_mat_0145_3_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_3, 136);  //B0(24-27) B1(24-27) B0(24-27) B1(24-27) B4(24-27) B5(24-27) B4(24-27) B5(24-27)
                const __m256i rhs_mat_2367_3_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_3, 136);  //B2(24-27) B3(24-27) B2(24-27) B3(24-27) B6(24-27) B7(24-27) B6(24-27) B7(24-27)

                // Shuffle pattern two - right side input

                const __m256i rhs_mat_0145_0_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_0, 221);  //B0(4-7) B1(4-7) B0(4-7) B1(4-7) B4(4-7) B5(4-7) B4(4-7) B5(4-7)
                const __m256i rhs_mat_2367_0_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_0, 221);  //B2(4-7) B3(4-7) B2(4-7) B3(4-7) B6(4-7) B7(4-7) B6(4-7) B7(4-7)

                const __m256i rhs_mat_0145_1_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_1, 221);  //B0(12-15) B1(12-15) B0(12-15) B1(12-15) B4(12-15) B5(12-15) B4(12-15) B5(12-15)
                const __m256i rhs_mat_2367_1_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_1, 221);  //B2(12-15) B3(12-15) B2(12-15) B3(12-15) B6(12-15) B7(12-15) B6(12-15) B7(12-15)

                const __m256i rhs_mat_0145_2_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_2, 221);  //B0(20-23) B1(20-23) B0(20-23) B1(20-23) B4(20-23) B5(20-23) B4(20-23) B5(20-23)
                const __m256i rhs_mat_2367_2_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_2, 221);  //B2(20-23) B3(20-23) B2(20-23) B3(20-23) B6(20-23) B7(20-23) B6(20-23) B7(20-23)

                const __m256i rhs_mat_0145_3_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_3, 221);  //B0(28-31) B1(28-31) B0(28-31) B1(28-31) B4(28-31) B5(28-31) B4(28-31) B5(28-31)
                const __m256i rhs_mat_2367_3_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_3, 221);  //B2(28-31) B3(28-31) B2(28-31) B3(28-31) B6(28-31) B7(28-31) B6(28-31) B7(28-31)

                // Scale values - Load the wight scale values of block_tx8
                __m256 col_scale_f32;
                if constexpr (
                        std::is_same_v<block_tx8, block_q4_0x8> ||
                        std::is_same_v<block_tx8, block_iq4_nlx8>) {
                    col_scale_f32 = GGML_F32Cx8_LOAD(b_ptr[b].d);
                }

                // Process LHS in groups of four
                for (int rp = 0; rp < 4; rp++) {
                    // Load the four blocks of quantized values interleaved with each other in chunks of eight - A0,A1,A2,A3
                    // Loaded as set of 128 bit vectors and repeated into a 256 bit vector
                    __m256i lhs_mat_0123_0 = _mm256_loadu_si256((const __m256i *)((a_ptrs[rp][b].qs)));
                    __m256i lhs_mat_01_0 = _mm256_permute2f128_si256(lhs_mat_0123_0, lhs_mat_0123_0, 0);
                    __m256i lhs_mat_23_0 = _mm256_permute2f128_si256(lhs_mat_0123_0, lhs_mat_0123_0, 17);
                    __m256i lhs_mat_0123_1 = _mm256_loadu_si256((const __m256i *)((a_ptrs[rp][b].qs + 32)));
                    __m256i lhs_mat_01_1 = _mm256_permute2f128_si256(lhs_mat_0123_1, lhs_mat_0123_1, 0);
                    __m256i lhs_mat_23_1 = _mm256_permute2f128_si256(lhs_mat_0123_1, lhs_mat_0123_1, 17);
                    __m256i lhs_mat_0123_2 = _mm256_loadu_si256((const __m256i *)((a_ptrs[rp][b].qs + 64)));
                    __m256i lhs_mat_01_2 = _mm256_permute2f128_si256(lhs_mat_0123_2, lhs_mat_0123_2, 0);
                    __m256i lhs_mat_23_2 = _mm256_permute2f128_si256(lhs_mat_0123_2, lhs_mat_0123_2, 17);
                    __m256i lhs_mat_0123_3 = _mm256_loadu_si256((const __m256i *)((a_ptrs[rp][b].qs + 96)));
                    __m256i lhs_mat_01_3 = _mm256_permute2f128_si256(lhs_mat_0123_3, lhs_mat_0123_3, 0);
                    __m256i lhs_mat_23_3 = _mm256_permute2f128_si256(lhs_mat_0123_3, lhs_mat_0123_3, 17);

                    // Shuffle pattern one - left side input
                    const __m256i lhs_mat_01_0_sp1 = _mm256_shuffle_epi32(lhs_mat_01_0, 160);  //A0(0-3) A0(0-3) A1(0-3) A1(0-3) A0(0-3) A0(0-3) A1(0-3) A1(0-3)
                    const __m256i lhs_mat_23_0_sp1 = _mm256_shuffle_epi32(lhs_mat_23_0, 160);  //A2(0-3) A2(0-3) A3(0-3) A3(0-3) A2(0-3) A2(0-3) A3(0-3) A3(0-3)

                    const __m256i lhs_mat_01_1_sp1 = _mm256_shuffle_epi32(lhs_mat_01_1, 160);  //A0(8-11) A0(8-11) A1(8-11) A1(8-11) A0(8-11) A0(8-11) A1(8-11) A1(8-11)
                    const __m256i lhs_mat_23_1_sp1 = _mm256_shuffle_epi32(lhs_mat_23_1, 160);  //A2(8-11) A2(8-11) A3(8-11) A3(8-11) A2(8-11) A2(8-11) A3(8-11) A3(8-11)

                    const __m256i lhs_mat_01_2_sp1 = _mm256_shuffle_epi32(lhs_mat_01_2, 160);  //A0(16-19) A0(16-19) A1(16-19) A1(16-19) A0(16-19) A0(16-19) A1(16-19) A1(16-19)
                    const __m256i lhs_mat_23_2_sp1 = _mm256_shuffle_epi32(lhs_mat_23_2, 160);  //A2(16-19) A2(16-19) A3(16-19) A3(16-19) A2(16-19) A2(16-19) A3(16-19) A3(16-19)

                    const __m256i lhs_mat_01_3_sp1 = _mm256_shuffle_epi32(lhs_mat_01_3, 160);  //A0(24-27) A0(24-27) A1(24-27) A1(24-27) A0(24-27) A0(24-27) A1(24-27) A1(24-27)
                    const __m256i lhs_mat_23_3_sp1 = _mm256_shuffle_epi32(lhs_mat_23_3, 160);  //A2(24-27) A2(24-27) A3(24-27) A3(24-27) A2(24-27) A2(24-27) A3(24-27) A3(24-27)

                    // Shuffle pattern two - left side input
                    const __m256i lhs_mat_01_0_sp2 = _mm256_shuffle_epi32(lhs_mat_01_0, 245);  //A0(4-7) A0(4-7) A1(4-7) A1(4-7) A0(4-7) A0(4-7) A1(4-7) A1(4-7)
                    const __m256i lhs_mat_23_0_sp2 = _mm256_shuffle_epi32(lhs_mat_23_0, 245);  //A2(4-7) A2(4-7) A3(4-7) A3(4-7) A2(4-7) A2(4-7) A3(4-7) A3(4-7)

                    const __m256i lhs_mat_01_1_sp2 = _mm256_shuffle_epi32(lhs_mat_01_1, 245);  //A0(12-15) A0(12-15) A1(12-15) A1(12-15) A0(12-15) A0(12-15) A1(12-15) A1(12-15)
                    const __m256i lhs_mat_23_1_sp2 = _mm256_shuffle_epi32(lhs_mat_23_1, 245);  //A2(12-15) A2(12-15) A3(12-15) A3(12-15) A2(12-15) A2(12-15) A3(12-15) A3(12-15)

                    const __m256i lhs_mat_01_2_sp2 = _mm256_shuffle_epi32(lhs_mat_01_2, 245);  //A0(20-23) A0(20-23) A1(20-23) A1(20-23) A0(20-23) A0(20-23) A1(20-23) A1(20-23)
                    const __m256i lhs_mat_23_2_sp2 = _mm256_shuffle_epi32(lhs_mat_23_2, 245);  //A2(20-23) A2(20-23) A3(20-23) A3(20-23) A2(20-23) A2(20-23) A3(20-23) A3(20-23)

                    const __m256i lhs_mat_01_3_sp2 = _mm256_shuffle_epi32(lhs_mat_01_3, 245);  //A0(28-31) A0(28-31) A1(28-31) A1(28-31) A0(28-31) A0(28-31) A1(28-31) A1(28-31)
                    const __m256i lhs_mat_23_3_sp2 = _mm256_shuffle_epi32(lhs_mat_23_3, 245);  //A2(28-31) A2(28-31) A3(28-31) A3(28-31) A2(28-31) A2(28-31) A3(28-31) A3(28-31)

                    // The values arranged in shuffle patterns are operated with dot product operation within 32 bit lane i.e corresponding bytes and multiplied and added into 32 bit integers within 32 bit lane
                    // Resembles MMLAs into 2x2 matrices in ARM Version
                    const __m256i zero = _mm256_setzero_si256();
                    __m256i iacc_mat_00_sp1 = mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_01_3_sp1, rhs_mat_0145_3_sp1), lhs_mat_01_2_sp1, rhs_mat_0145_2_sp1), lhs_mat_01_1_sp1, rhs_mat_0145_1_sp1), lhs_mat_01_0_sp1, rhs_mat_0145_0_sp1);
                    __m256i iacc_mat_01_sp1 = mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_01_3_sp1, rhs_mat_2367_3_sp1), lhs_mat_01_2_sp1, rhs_mat_2367_2_sp1), lhs_mat_01_1_sp1, rhs_mat_2367_1_sp1), lhs_mat_01_0_sp1, rhs_mat_2367_0_sp1);
                    __m256i iacc_mat_10_sp1 = mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_23_3_sp1, rhs_mat_0145_3_sp1), lhs_mat_23_2_sp1, rhs_mat_0145_2_sp1), lhs_mat_23_1_sp1, rhs_mat_0145_1_sp1), lhs_mat_23_0_sp1, rhs_mat_0145_0_sp1);
                    __m256i iacc_mat_11_sp1 = mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_23_3_sp1, rhs_mat_2367_3_sp1), lhs_mat_23_2_sp1, rhs_mat_2367_2_sp1), lhs_mat_23_1_sp1, rhs_mat_2367_1_sp1), lhs_mat_23_0_sp1, rhs_mat_2367_0_sp1);
                    __m256i iacc_mat_00_sp2 = mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_01_3_sp2, rhs_mat_0145_3_sp2), lhs_mat_01_2_sp2, rhs_mat_0145_2_sp2), lhs_mat_01_1_sp2, rhs_mat_0145_1_sp2), lhs_mat_01_0_sp2, rhs_mat_0145_0_sp2);
                    __m256i iacc_mat_01_sp2 = mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_01_3_sp2, rhs_mat_2367_3_sp2), lhs_mat_01_2_sp2, rhs_mat_2367_2_sp2), lhs_mat_01_1_sp2, rhs_mat_2367_1_sp2), lhs_mat_01_0_sp2, rhs_mat_2367_0_sp2);
                    __m256i iacc_mat_10_sp2 = mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_23_3_sp2, rhs_mat_0145_3_sp2), lhs_mat_23_2_sp2, rhs_mat_0145_2_sp2), lhs_mat_23_1_sp2, rhs_mat_0145_1_sp2), lhs_mat_23_0_sp2, rhs_mat_0145_0_sp2);
                    __m256i iacc_mat_11_sp2 = mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_23_3_sp2, rhs_mat_2367_3_sp2), lhs_mat_23_2_sp2, rhs_mat_2367_2_sp2), lhs_mat_23_1_sp2, rhs_mat_2367_1_sp2), lhs_mat_23_0_sp2, rhs_mat_2367_0_sp2);

                    // Output of both shuffle patterns are added in order to sum dot product outputs of all 32 values in block
                    __m256i iacc_mat_00 = _mm256_add_epi32(iacc_mat_00_sp1, iacc_mat_00_sp2);
                    __m256i iacc_mat_01 = _mm256_add_epi32(iacc_mat_01_sp1, iacc_mat_01_sp2);
                    __m256i iacc_mat_10 = _mm256_add_epi32(iacc_mat_10_sp1, iacc_mat_10_sp2);
                    __m256i iacc_mat_11 = _mm256_add_epi32(iacc_mat_11_sp1, iacc_mat_11_sp2);

                    // Straighten out to make 4 row vectors
                    __m256i iacc_row_0 = _mm256_blend_epi32(iacc_mat_00, _mm256_shuffle_epi32(iacc_mat_01, 78), 204);
                    __m256i iacc_row_1 = _mm256_blend_epi32(_mm256_shuffle_epi32(iacc_mat_00, 78), iacc_mat_01, 204);
                    __m256i iacc_row_2 = _mm256_blend_epi32(iacc_mat_10, _mm256_shuffle_epi32(iacc_mat_11, 78), 204);
                    __m256i iacc_row_3 = _mm256_blend_epi32(_mm256_shuffle_epi32(iacc_mat_10, 78), iacc_mat_11, 204);

                    // Load the scale(d) values for all the 4 Q8_0 blocks and repeat it across lanes
                    const __m256 row_scale_f32 = GGML_F32Cx8_REPEAT_LOAD(a_ptrs[rp][b].d, loadMask);

                    // Multiply with appropiate scales and accumulate
                    acc_rows[rp * 4] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_0), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 0)), acc_rows[rp * 4]);
                    acc_rows[rp * 4 + 1] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_1), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 85)), acc_rows[rp * 4 + 1]);
                    acc_rows[rp * 4 + 2] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_2), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 170)), acc_rows[rp * 4 + 2]);
                    acc_rows[rp * 4 + 3] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_3), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32,  255)), acc_rows[rp * 4 + 3]);
                }
            }

            // Store the accumulated values
            for (int i = 0; i < 16; i++) {
                _mm256_storeu_ps((float *)(s + ((y * 4 + i) * bs + x * 8)), acc_rows[i]);
            }
        }
    }

    // Take a block_q8_0x4 structures at each pass of the loop and perform dot product operation
    for (; y < nr / 4; y ++) {
        const block_q8_0x4 * a_ptr = a_ptr_start + (y * nb);

        // Load the eight blocks of quantized values interleaved with each other in chunks of eight - B0,B1 ....B6,B7
        for (int64_t x = xstart; x < nc / 8; x++) {
            const block_tx8 * b_ptr = b_ptr_start + (x * b_nb);

            // Master FP accumulators
            __m256 acc_rows[4];
            for (int i = 0; i < 4; i++) {
                acc_rows[i] = _mm256_setzero_ps();
            }

            for (int64_t b = 0; b < nb; b++) {
                // Load the eight block_q8_0 quantized values interleaved with each other in chunks of eight - B0,B1 ....B6,B7
                const __m256i rhs_raw_mat_0123_0 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs));
                const __m256i rhs_raw_mat_4567_0 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 32));
                const __m256i rhs_raw_mat_0123_1 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 64));
                const __m256i rhs_raw_mat_4567_1 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 96));

                // Save the values in the following vectors in the formats B0B1B4B5, B2B3B6B7 for further processing and storing of valuess
                const __m256i rhs_raw_mat_0145_0 = _mm256_blend_epi32(rhs_raw_mat_0123_0, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_0, requiredOrder), 240);
                const __m256i rhs_raw_mat_2367_0 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_0, requiredOrder), rhs_raw_mat_4567_0, 240);
                const __m256i rhs_raw_mat_0145_1 = _mm256_blend_epi32(rhs_raw_mat_0123_1, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_1, requiredOrder), 240);
                const __m256i rhs_raw_mat_2367_1 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_1, requiredOrder), rhs_raw_mat_4567_1, 240);

                // 4-bit -> 8-bit - Sign is maintained
                const __m256i rhs_mat_0145_0 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(rhs_raw_mat_0145_0, m4b));  //B0(0-7) B1(0-7) B4(0-7) B5(0-7)
                const __m256i rhs_mat_2367_0 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(rhs_raw_mat_2367_0, m4b));  //B2(0-7) B3(0-7) B6(0-7) B7(0-7)

                const __m256i rhs_mat_0145_1 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(rhs_raw_mat_0145_1, m4b));  //B0(8-15) B1(8-15) B4(8-15) B5(8-15)
                const __m256i rhs_mat_2367_1 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(rhs_raw_mat_2367_1, m4b));  //B2(8-15) B3(8-15) B6(8-15) B7(8-15)

                const __m256i rhs_mat_0145_2 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_0, 4), m4b));  //B0(16-23) B1(16-23) B4(16-23) B5(16-23)
                const __m256i rhs_mat_2367_2 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_0, 4), m4b));  //B2(16-23) B3(16-23) B6(16-23) B7(16-23)

                const __m256i rhs_mat_0145_3 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_1, 4), m4b));  //B0(24-31) B1(24-31) B4(24-31) B5(24-31)
                const __m256i rhs_mat_2367_3 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_1, 4), m4b));  //B2(24-31) B3(24-31) B6(24-31) B7(24-31)

                // Shuffle pattern one - right side input
                const __m256i rhs_mat_0145_0_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_0, 136);  //B0(0-3) B1(0-3) B0(0-3) B1(0-3) B4(0-3) B5(0-3) B4(0-3) B5(0-3)
                const __m256i rhs_mat_2367_0_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_0, 136);  //B2(0-3) B3(0-3) B2(0-3) B3(0-3) B6(0-3) B7(0-3) B6(0-3) B7(0-3)

                const __m256i rhs_mat_0145_1_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_1, 136);  //B0(8-11) B1(8-11) B0(8-11) B1(8-11) B4(8-11) B5(8-11) B4(8-11) B5(8-11)
                const __m256i rhs_mat_2367_1_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_1, 136);  //B2(8-11) B3(8-11) B2(8-11) B3(8-11) B6(8-11) B7(8-11) B6(8-11) B7(8-11)

                const __m256i rhs_mat_0145_2_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_2, 136);  //B0(16-19) B1(16-19) B0(16-19) B1(16-19) B4(16-19) B5(16-19) B4(16-19) B5(16-19)
                const __m256i rhs_mat_2367_2_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_2, 136);  //B2(16-19) B3(16-19) B2(16-19) B3(16-19) B6(16-19) B7(16-19) B6(16-19) B7(16-19)

                const __m256i rhs_mat_0145_3_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_3, 136);  //B0(24-27) B1(24-27) B0(24-27) B1(24-27) B4(24-27) B5(24-27) B4(24-27) B5(24-27)
                const __m256i rhs_mat_2367_3_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_3, 136);  //B2(24-27) B3(24-27) B2(24-27) B3(24-27) B6(24-27) B7(24-27) B6(24-27) B7(24-27)

                // Shuffle pattern two - right side input

                const __m256i rhs_mat_0145_0_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_0, 221);  //B0(4-7) B1(4-7) B0(4-7) B1(4-7) B4(4-7) B5(4-7) B4(4-7) B5(4-7)
                const __m256i rhs_mat_2367_0_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_0, 221);  //B2(4-7) B3(4-7) B2(4-7) B3(4-7) B6(4-7) B7(4-7) B6(4-7) B7(4-7)

                const __m256i rhs_mat_0145_1_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_1, 221);  //B0(12-15) B1(12-15) B0(12-15) B1(12-15) B4(12-15) B5(12-15) B4(12-15) B5(12-15)
                const __m256i rhs_mat_2367_1_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_1, 221);  //B2(12-15) B3(12-15) B2(12-15) B3(12-15) B6(12-15) B7(12-15) B6(12-15) B7(12-15)

                const __m256i rhs_mat_0145_2_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_2, 221);  //B0(20-23) B1(20-23) B0(20-23) B1(20-23) B4(20-23) B5(20-23) B4(20-23) B5(20-23)
                const __m256i rhs_mat_2367_2_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_2, 221);  //B2(20-23) B3(20-23) B2(20-23) B3(20-23) B6(20-23) B7(20-23) B6(20-23) B7(20-23)

                const __m256i rhs_mat_0145_3_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_3, 221);  //B0(28-31) B1(28-31) B0(28-31) B1(28-31) B4(28-31) B5(28-31) B4(28-31) B5(28-31)
                const __m256i rhs_mat_2367_3_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_3, 221);  //B2(28-31) B3(28-31) B2(28-31) B3(28-31) B6(28-31) B7(28-31) B6(28-31) B7(28-31)

                // Scale values - Load the wight scale values of block_tx8
                __m256 col_scale_f32;
                if constexpr (
                        std::is_same_v<block_tx8, block_q4_0x8> ||
                        std::is_same_v<block_tx8, block_iq4_nlx8>) {
                    col_scale_f32 = GGML_F32Cx8_LOAD(b_ptr[b].d);
                }

                // Load the four blocks of quantized values interleaved with each other in chunks of eight - A0,A1,A2,A3
                // Loaded as set of 128 bit vectors and repeated into a 256 bit vector
                __m256i lhs_mat_0123_0 = _mm256_loadu_si256((const __m256i *)((a_ptr[b].qs)));
                __m256i lhs_mat_01_0 = _mm256_permute2f128_si256(lhs_mat_0123_0, lhs_mat_0123_0, 0);
                __m256i lhs_mat_23_0 = _mm256_permute2f128_si256(lhs_mat_0123_0, lhs_mat_0123_0, 17);
                __m256i lhs_mat_0123_1 = _mm256_loadu_si256((const __m256i *)((a_ptr[b].qs + 32)));
                __m256i lhs_mat_01_1 = _mm256_permute2f128_si256(lhs_mat_0123_1, lhs_mat_0123_1, 0);
                __m256i lhs_mat_23_1 = _mm256_permute2f128_si256(lhs_mat_0123_1, lhs_mat_0123_1, 17);
                __m256i lhs_mat_0123_2 = _mm256_loadu_si256((const __m256i *)((a_ptr[b].qs + 64)));
                __m256i lhs_mat_01_2 = _mm256_permute2f128_si256(lhs_mat_0123_2, lhs_mat_0123_2, 0);
                __m256i lhs_mat_23_2 = _mm256_permute2f128_si256(lhs_mat_0123_2, lhs_mat_0123_2, 17);
                __m256i lhs_mat_0123_3 = _mm256_loadu_si256((const __m256i *)((a_ptr[b].qs + 96)));
                __m256i lhs_mat_01_3 = _mm256_permute2f128_si256(lhs_mat_0123_3, lhs_mat_0123_3, 0);
                __m256i lhs_mat_23_3 = _mm256_permute2f128_si256(lhs_mat_0123_3, lhs_mat_0123_3, 17);

                // Shuffle pattern one - left side input

                const __m256i lhs_mat_01_0_sp1 = _mm256_shuffle_epi32(lhs_mat_01_0, 160);  //A0(0-3) A0(0-3) A1(0-3) A1(0-3) A0(0-3) A0(0-3) A1(0-3) A1(0-3)
                const __m256i lhs_mat_23_0_sp1 = _mm256_shuffle_epi32(lhs_mat_23_0, 160);  //A2(0-3) A2(0-3) A3(0-3) A3(0-3) A2(0-3) A2(0-3) A3(0-3) A3(0-3)

                const __m256i lhs_mat_01_1_sp1 = _mm256_shuffle_epi32(lhs_mat_01_1, 160);  //A0(8-11) A0(8-11) A1(8-11) A1(8-11) A0(8-11) A0(8-11) A1(8-11) A1(8-11)
                const __m256i lhs_mat_23_1_sp1 = _mm256_shuffle_epi32(lhs_mat_23_1, 160);  //A2(8-11) A2(8-11) A3(8-11) A3(8-11) A2(8-11) A2(8-11) A3(8-11) A3(8-11)

                const __m256i lhs_mat_01_2_sp1 = _mm256_shuffle_epi32(lhs_mat_01_2, 160);  //A0(16-19) A0(16-19) A1(16-19) A1(16-19) A0(16-19) A0(16-19) A1(16-19) A1(16-19)
                const __m256i lhs_mat_23_2_sp1 = _mm256_shuffle_epi32(lhs_mat_23_2, 160);  //A2(16-19) A2(16-19) A3(16-19) A3(16-19) A2(16-19) A2(16-19) A3(16-19) A3(16-19)

                const __m256i lhs_mat_01_3_sp1 = _mm256_shuffle_epi32(lhs_mat_01_3, 160);  //A0(24-27) A0(24-27) A1(24-27) A1(24-27) A0(24-27) A0(24-27) A1(24-27) A1(24-27)
                const __m256i lhs_mat_23_3_sp1 = _mm256_shuffle_epi32(lhs_mat_23_3, 160);  //A2(24-27) A2(24-27) A3(24-27) A3(24-27) A2(24-27) A2(24-27) A3(24-27) A3(24-27)

                // Shuffle pattern two - left side input

                const __m256i lhs_mat_01_0_sp2 = _mm256_shuffle_epi32(lhs_mat_01_0, 245);  //A0(4-7) A0(4-7) A1(4-7) A1(4-7) A0(4-7) A0(4-7) A1(4-7) A1(4-7)
                const __m256i lhs_mat_23_0_sp2 = _mm256_shuffle_epi32(lhs_mat_23_0, 245);  //A2(4-7) A2(4-7) A3(4-7) A3(4-7) A2(4-7) A2(4-7) A3(4-7) A3(4-7)

                const __m256i lhs_mat_01_1_sp2 = _mm256_shuffle_epi32(lhs_mat_01_1, 245);  //A0(12-15) A0(12-15) A1(12-15) A1(12-15) A0(12-15) A0(12-15) A1(12-15) A1(12-15)
                const __m256i lhs_mat_23_1_sp2 = _mm256_shuffle_epi32(lhs_mat_23_1, 245);  //A2(12-15) A2(12-15) A3(12-15) A3(12-15) A2(12-15) A2(12-15) A3(12-15) A3(12-15)

                const __m256i lhs_mat_01_2_sp2 = _mm256_shuffle_epi32(lhs_mat_01_2, 245);  //A0(20-23) A0(20-23) A1(20-23) A1(20-23) A0(20-23) A0(20-23) A1(20-23) A1(20-23)
                const __m256i lhs_mat_23_2_sp2 = _mm256_shuffle_epi32(lhs_mat_23_2, 245);  //A2(20-23) A2(20-23) A3(20-23) A3(20-23) A2(20-23) A2(20-23) A3(20-23) A3(20-23)

                const __m256i lhs_mat_01_3_sp2 = _mm256_shuffle_epi32(lhs_mat_01_3, 245);  //A0(28-31) A0(28-31) A1(28-31) A1(28-31) A0(28-31) A0(28-31) A1(28-31) A1(28-31)
                const __m256i lhs_mat_23_3_sp2 = _mm256_shuffle_epi32(lhs_mat_23_3, 245);  //A2(28-31) A2(28-31) A3(28-31) A3(28-31) A2(28-31) A2(28-31) A3(28-31) A3(28-31)

                // The values arranged in shuffle patterns are operated with dot product operation within 32 bit lane i.e corresponding bytes and multiplied and added into 32 bit integers within 32 bit lane
                // Resembles MMLAs into 2x2 matrices in ARM Version
                const __m256i zero = _mm256_setzero_si256();
                __m256i iacc_mat_00_sp1 = mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_01_3_sp1, rhs_mat_0145_3_sp1), lhs_mat_01_2_sp1, rhs_mat_0145_2_sp1), lhs_mat_01_1_sp1, rhs_mat_0145_1_sp1), lhs_mat_01_0_sp1, rhs_mat_0145_0_sp1);
                __m256i iacc_mat_01_sp1 = mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_01_3_sp1, rhs_mat_2367_3_sp1), lhs_mat_01_2_sp1, rhs_mat_2367_2_sp1), lhs_mat_01_1_sp1, rhs_mat_2367_1_sp1), lhs_mat_01_0_sp1, rhs_mat_2367_0_sp1);
                __m256i iacc_mat_10_sp1 = mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_23_3_sp1, rhs_mat_0145_3_sp1), lhs_mat_23_2_sp1, rhs_mat_0145_2_sp1), lhs_mat_23_1_sp1, rhs_mat_0145_1_sp1), lhs_mat_23_0_sp1, rhs_mat_0145_0_sp1);
                __m256i iacc_mat_11_sp1 = mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_23_3_sp1, rhs_mat_2367_3_sp1), lhs_mat_23_2_sp1, rhs_mat_2367_2_sp1), lhs_mat_23_1_sp1, rhs_mat_2367_1_sp1), lhs_mat_23_0_sp1, rhs_mat_2367_0_sp1);
                __m256i iacc_mat_00_sp2 = mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_01_3_sp2, rhs_mat_0145_3_sp2), lhs_mat_01_2_sp2, rhs_mat_0145_2_sp2), lhs_mat_01_1_sp2, rhs_mat_0145_1_sp2), lhs_mat_01_0_sp2, rhs_mat_0145_0_sp2);
                __m256i iacc_mat_01_sp2 = mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_01_3_sp2, rhs_mat_2367_3_sp2), lhs_mat_01_2_sp2, rhs_mat_2367_2_sp2), lhs_mat_01_1_sp2, rhs_mat_2367_1_sp2), lhs_mat_01_0_sp2, rhs_mat_2367_0_sp2);
                __m256i iacc_mat_10_sp2 = mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_23_3_sp2, rhs_mat_0145_3_sp2), lhs_mat_23_2_sp2, rhs_mat_0145_2_sp2), lhs_mat_23_1_sp2, rhs_mat_0145_1_sp2), lhs_mat_23_0_sp2, rhs_mat_0145_0_sp2);
                __m256i iacc_mat_11_sp2 = mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_23_3_sp2, rhs_mat_2367_3_sp2), lhs_mat_23_2_sp2, rhs_mat_2367_2_sp2), lhs_mat_23_1_sp2, rhs_mat_2367_1_sp2), lhs_mat_23_0_sp2, rhs_mat_2367_0_sp2);

                // Output of both shuffle patterns are added in order to sum dot product outputs of all 32 values in block
                __m256i iacc_mat_00 = _mm256_add_epi32(iacc_mat_00_sp1, iacc_mat_00_sp2);
                __m256i iacc_mat_01 = _mm256_add_epi32(iacc_mat_01_sp1, iacc_mat_01_sp2);
                __m256i iacc_mat_10 = _mm256_add_epi32(iacc_mat_10_sp1, iacc_mat_10_sp2);
                __m256i iacc_mat_11 = _mm256_add_epi32(iacc_mat_11_sp1, iacc_mat_11_sp2);


                // Straighten out to make 4 row vectors
                __m256i iacc_row_0 = _mm256_blend_epi32(iacc_mat_00, _mm256_shuffle_epi32(iacc_mat_01, 78), 204);
                __m256i iacc_row_1 = _mm256_blend_epi32(_mm256_shuffle_epi32(iacc_mat_00, 78), iacc_mat_01, 204);
                __m256i iacc_row_2 = _mm256_blend_epi32(iacc_mat_10, _mm256_shuffle_epi32(iacc_mat_11, 78), 204);
                __m256i iacc_row_3 = _mm256_blend_epi32(_mm256_shuffle_epi32(iacc_mat_10, 78), iacc_mat_11, 204);

                // Load the scale(d) values for all the 4 Q8_0 blocks and repeat it across lanes
                const __m256 row_scale_f32 = GGML_F32Cx8_REPEAT_LOAD(a_ptr[b].d, loadMask);

                // Multiply with appropiate scales and accumulate
                acc_rows[0] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_0), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 0)), acc_rows[0]);
                acc_rows[1] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_1), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 85)), acc_rows[1]);
                acc_rows[2] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_2), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 170)), acc_rows[2]);
                acc_rows[3] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_3), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 255)), acc_rows[3]);
            }

            // Store the accumulated values
            for (int i = 0; i < 4; i++) {
                _mm256_storeu_ps((float *)(s + ((y * 4 + i) * bs + x * 8)), acc_rows[i]);
            }
        }
    }
}
```
_wip_

### test-backend-ops testing repack correctness
The goal is to allow `test-backend-ops` to compare the implementation of weight
repacking with the extra buffer type, to the base implementation with the
standard buffer type, to verify the correctness of the extra buffer type
implementation. 

Repack is not an operation like the other operations that are tested by
`test-backend-ops` so adding a test_case for it does not sound correct...but
it is part of a matrix multiplication operation, so perhaps adding a test that
subclasses `test_mul_mat_id` would make sense.

```console
$ ./build/bin/test-backend-ops -o "MUL_MAT(type_a=q4_0,type_b=f32,m=16,n=1,k=256,bs=[1,1],nr=[1,1],per=[0,1,2,3],v=0,o=1)"
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 4070, compute capability 8.9, VMM: yes
register_backend: registered backend CUDA (1 devices)
register_device: registered device CUDA0 (NVIDIA GeForce RTX 4070)
register_backend: registered backend CPU (1 devices)
register_device: registered device CPU (12th Gen Intel(R) Core(TM) i7-1260P)
load_backend: failed to find ggml_backend_init in /home/danbev/work/ai/llama.cpp-debug/build/bin/libggml-cuda.so
load_backend: failed to find ggml_backend_init in /home/danbev/work/ai/llama.cpp-debug/build/bin/libggml-cpu.so
Testing 2 devices

Backend 1/2: CUDA0
  Device description: NVIDIA GeForce RTX 4070
  Device memory: 11903 MB (11743 MB free)

  MUL_MAT(type_a=q4_0,type_b=f32,m=16,n=1,k=256,bs=[1,1],nr=[1,1],per=[0,1,2,3],v=0,o=1): OK
  MUL_MAT(type_a=q4_0,type_b=f32,m=16,n=1,k=256,bs=[1,1],nr=[1,1],per=[0,1,2,3],v=0,o=1): OK
  14471/14471 tests passed
  Backend CUDA0: OK
Backend 2/2: CPU
  Skipping CPU backend
2/2 backends passed
OK
```

```c++
struct test_mul_mat : public test_case {
    const ggml_type type_a;
    const ggml_type type_b;
    const int64_t m;
    const int64_t n;
    const int64_t k;
    const std::array<int64_t, 2> bs;  // dims 3 and 4
    const std::array<int64_t, 2> nr;  // repeat in dims 3 and 4
    const std::array<int64_t, 4> per; // permutation of dimensions
    const bool v; // whether a and b are non-contiguous views
    const uint32_t o; // number of outputs

    test_mul_mat(ggml_type type_a = GGML_TYPE_F32, ggml_type type_b = GGML_TYPE_F32,
            int64_t m = 32, int64_t n = 32, int64_t k = 32,
            std::array<int64_t, 2> bs = {10, 10},
            std::array<int64_t, 2> nr = {2, 2},
            std::array<int64_t, 4> per = {0, 1, 2, 3},
            bool v = false, uint32_t o = 1)
        : type_a(type_a), type_b(type_b), m(m), n(n), k(k), bs(bs), nr(nr), per(per), v(v), o(o) {}
```
The types are the types for the A and B matrices. And m, n, k are the dimensions,
A is [mxk] and B is [kxn]. The output matrix C is [mxn].
bs is the batch size for batched matrix multiplication, for 2D matrices it is {1,1}.
nr is the number of repeats in each batch dimension, for 2D matrices it is {1,1}.
per is the permutation of the 4 dimensions (m,n,batch0,batch1). 0, 1, 2, 3 is just
the normal order.
v is whether A and B are non-contiguous views.
o is the number of outputs to test generate.

To be able to debug this I've replaced (I renamed the original): 
```c++
static std::vector<std::unique_ptr<test_case>> make_test_cases_eval() {
    std::vector<std::unique_ptr<test_case>> test_cases;
    std::default_random_engine rng(0);

    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q4_0, GGML_TYPE_F32, 16,  1, 256, {1, 1}, {1, 1}));

    return test_cases;
}
```
This is because it was difficult to set breakpoints as all test cases are processed
and then filtered.

Inspecting the tensors, skipping the first sentinel tensor, we have:
```console
(gdb) p *t
$6 = {type = GGML_TYPE_Q4_0, buffer = 0x0, ne = {256, 16, 1, 1},
nb = {18, 144, 2304, 2304}, op = GGML_OP_NONE, op_params = { 0 <repeats 16 times>},
flags = 0, src = {0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0}, view_src = 0x0, view_offs = 0,
data = 0x0, name = "a", '\000' <repeats 62 times>, extra = 0x0, padding = "\000\000\000\000\000\000\000"}
```

But there is no extra buffer allocated for repack so that will not be used here.
So why does this not get applied for the test. The issue is that the extra
buffer types are not used in the test as the are when a model is loaded like
we saw above.

We have to add something like the following (just for testing at this stage to
force something to work):
```c++
        std::vector<ggml_backend_buffer_t> buffers;
        std::vector<ggml_backend_buffer_type_t> extra_buft_list;
        auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        auto * cpu_reg = ggml_backend_dev_backend_reg(cpu_dev);
        auto get_extra_bufts_fn = (ggml_backend_dev_get_extra_bufts_t)
            ggml_backend_reg_get_proc_address(cpu_reg, "ggml_backend_dev_get_extra_bufts");

        if (get_extra_bufts_fn) {
            ggml_backend_buffer_type_t * extra_bufts = get_extra_bufts_fn(cpu_dev);
            while (extra_bufts && *extra_bufts) {
                extra_buft_list.push_back(*extra_bufts);
                ++extra_bufts;
            }
        }

        // Try to find repack buffer type among the extra buffer types
        ggml_backend_buffer_type_t repack_buft = nullptr;
        for (auto buft : extra_buft_list) {
            const char* buft_name = ggml_backend_buft_name(buft);
            if (buft_name && strstr(buft_name, "CPU_REPACK")) {
                repack_buft = buft;
                break;
            }
        }
        //ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, repack_buft);
```
This by it self does not work as this will cause a segment fault in repack for
the sentinel tensor:
```console
Thread 1 "test-backend-op" received signal SIGSEGV, Segmentation fault.
0x00007ffff7852616 in ggml_backend_cpu_repack_buffer_set_tensor (buffer=0x55555567c670, tensor=0x5555556914d0, data=0x5555556794a0, 
    offset=0, size=4096) at /home/danbev/work/ai/llama.cpp-debug/ggml/src/ggml-cpu/repack.cpp:1885
1885	    auto OK            = tensor_traits->repack(tensor, data, size);
(gdb) p *tensor
$1 = {type = GGML_TYPE_F32, buffer = 0x55555567c670, ne = {1024, 1, 1, 1}, nb = {4, 4096, 4096, 4096}, op = GGML_OP_NONE, 
op_params = {0 <repeats 16 times>}, flags = 0, src = {0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0}, view_src = 0x0, 
view_offs = 0, data = 0x55555569ccc0, name = "sent_0", '\000' <repeats 57 times>,
extra = 0x0, padding = "\000\000\000\000\000\000\000"}
```
We can see that this tensor does not have an extra buffer set. We can work around
that with a check perhaps.

But back to the test and looking at how the buffer is used, the graph is forward
expanded:
```c++
        // build graph
        ggml_build_forward_expand(gf, out);
```
Then the tensors data are set:
```c++
        // randomize tensors
        initialize_tensors(ctx);
```
This will call the repack buffer types set_tensor function if that buffer type
was used to create the backend buffer.

Next the callback user data type is defined and one instance created:
```c++
        // compare
        struct callback_userdata {
            bool   ok;
            double max_err;
            ggml_backend_t backend1;
            ggml_backend_t backend2;
        };

        callback_userdata ud {
            true,
            max_nmse_err(),
            backend1,
            backend2
        };
```
The callback itself is a lambda that looks like this:
```c++
        auto callback = [](int index, ggml_tensor * t1, ggml_tensor * t2, void * user_data) -> bool {
            callback_userdata * ud = (callback_userdata *) user_data;
            const char * bn1 = ggml_backend_name(ud->backend1);
            const char * bn2 = ggml_backend_name(ud->backend2);

            if (t1->op == GGML_OP_NONE) {
                // sentinels must be unchanged
                std::vector<uint8_t> t1_data(ggml_nbytes(t1));
                std::vector<uint8_t> t2_data(ggml_nbytes(t2));
                ggml_backend_tensor_get(t1, t1_data.data(), 0, ggml_nbytes(t1));
                ggml_backend_tensor_get(t2, t2_data.data(), 0, ggml_nbytes(t2));

                if (memcmp(t1_data.data(), t2_data.data(), ggml_nbytes(t1)) != 0) {
                    printf("sentinel mismatch: %s ", t1->name);
                    ud->ok = false;
                    return true;
                }
            }

            // convert to float for comparision as the data format migth be
            // in different formats, like normalizing.
            std::vector<float> f1 = tensor_to_float(t1);
            std::vector<float> f2 = tensor_to_float(t2);

            for (size_t i = 0; i < f1.size(); i++) {
                // check for nans
                if (std::isnan(f1[i]) || std::isnan(f2[i])) {
                    printf("[%s] NaN at index %zu (%s=%f %s=%f) ", ggml_op_desc(t1), i, bn1, f1[i], bn2, f2[i]);
                    ud->ok = false;
                    return true;
                }
                // check for infs: both must be inf of the same sign, or both must be finite
                if (isinf_or_max(f1[i]) || isinf_or_max(f2[i])) {
                    if (isinf_or_max(f1[i]) && isinf_or_max(f2[i])) {
                        if (std::signbit(f1[i]) != std::signbit(f2[i])) {
                            printf("[%s] inf sign mismatch: %s=%f %s=%f ", ggml_op_desc(t1), bn1, f1[i], bn2, f2[i]);
                            ud->ok = false;
                            return true;
                        }
                    } else {
                        printf("[%s] inf mismatch: %s=%f %s=%f ", ggml_op_desc(t1), bn1, f1[i], bn2, f2[i]);
                        ud->ok = false;
                        return true;
                    }
                }
            }

            double err = nmse(f1.data(), f2.data(), f1.size());
            if (err > ud->max_err) {
                printf("[%s] NMSE = %.9f > %.9f ", ggml_op_desc(t1), err, ud->max_err);
                //for (int i = 0; i < (int) f1.size(); i++) {
                //    printf("%5d %9.6f %9.6f, diff = %9.6f\n", i, f1[i], f2[i], f1[i] - f2[i]);
                //}
                //printf("\n");
                //exit(1);
                ud->ok = false;
            }
            return true;

            GGML_UNUSED(index);
        };
        const bool cmp_ok = ggml_backend_compare_graph_backend(backend1, backend2, gf, callback, &ud, run_whole_graph() ? out : nullptr);
```
The `callback` is passed into `ggml_backend_compare_graph_backend` along with the
two backends, the graph and a test_node which is either the output node or null
(and is null in this case):
```c++
bool ggml_backend_compare_graph_backend(ggml_backend_t backend1, ggml_backend_t backend2, struct ggml_cgraph * graph, ggml_backend_eval_callback callback, void * user_data, struct ggml_tensor * test_node) {
    struct ggml_backend_graph_copy copy = ggml_backend_graph_copy(backend2, graph);
    if (copy.buffer == NULL) {
        return false;
    }

    struct ggml_cgraph * g1 = graph;
    struct ggml_cgraph * g2 = copy.graph;
    ...
```
So, we only have one graph, but we want to compare the result of running this
graph on two different backends. So we create a copy of the graph.

Next, we 
```c++
    ...
    } else {
        for (int i = 0; i < g1->n_nodes; i++) {
            struct ggml_tensor * t1 = g1->nodes[i]; // node we want to test
            struct ggml_tensor * t2 = g2->nodes[i]; // node 

            assert(t1->op == t2->op && ggml_are_same_layout(t1, t2));

            // Create subgraphs containing only one operation
            struct ggml_cgraph g1v = ggml_graph_view(g1, i, i + 1); // use node i only
            struct ggml_cgraph g2v = ggml_graph_view(g2, i, i + 1); // use node i only

            ggml_backend_graph_compute(backend1, &g1v);
            ggml_backend_graph_compute(backend2, &g2v);

            if (ggml_is_view_op(t1->op)) {
                continue;
            }

            // compare results, calculate rms etc. t1 and t1 now contain the
            // result of this specific opertion
            if (!callback(i, t1, t2, user_data)) {
                break;
            }
        }
```
So this is going compute one node at a time.  Now recall that ggml will sort of 
compute backwards using "dependency resolution", like the output node is what
will be computed first in this case:
```console
(gdb) p *t1
$49 = {type = GGML_TYPE_F32, buffer = 0x55555567c670, ne = {16, 1, 1, 1},
nb = {4, 64, 64, 64}, op = GGML_OP_MUL_MAT, op_params = { 0 <repeats 16 times>}, flags = 0,
src = {0x555555691640, 0x555555691920, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0}, 
view_src = 0x0, view_offs = 0, data = 0x5555556a09c0,
name = "out", '\000' <repeats 60 times>, extra = 0x0, padding = "\000\000\000\000\000\000\000"}
```
And all the nodes in the graph look like this:
```console
(gdb) p g1->n_nodes
$51 = 5
(gdb) p g1->nodes[0]->name
$53 = "out", '\000' <repeats 60 times>
(gdb) p g1->nodes[1]->name
$54 = "sent_0", '\000' <repeats 57 times>
(gdb) p g1->nodes[2]->name
$55 = "sent_1", '\000' <repeats 57 times>
(gdb) p g1->nodes[3]->name
$56 = "sent_2", '\000' <repeats 57 times>
(gdb) p g1->nodes[4]->name
$57 = "sent_3", '\000' <repeats 57 times>
```
Notice that this tensor has input tensors which will be computed first which
is what I mean be dependency resolution and kind of working backwards.
The above will then create a graph view of the single operation MUL_MAT and then
compute it for both backends. After result of these operations will be in t1 and
t2 and the callback will be called to compare them.

In the callback we can inspect the backends being compared:
```console
(gdb) p ggml_backend_dev_name(ud->backend2->device)
$11 = 0x7ffff7542487 "CPU-ref"
(gdb) p ggml_backend_dev_name(ud->backend1->device)
$12 = 0x7ffff7942487 "CPU-alderlake"
```
So the tensor will be converted to float vectors for comparison as the output/result:
tensors can be in different formats and this is a way of normalizing them for
comparison:
```c++
            std::vector<float> f1 = tensor_to_float(t1);
            std::vector<float> f2 = tensor_to_float(t2);
```
This should not pose a problem for the repacking as the resulting tensor will
not be in repacked format, that is only the input tensors that are repacked.

So like we mentioned before, we need to somehow use the repack buffer type
when allocating the backend buffer for the tensor so that the repack set_tensor
is called and the repacking happends:
```c++
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, repack_buft);
        //ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend1);
```
```console
(gdb) p ggml_backend_buffer_name(buf)
$3 = 0x7ffff794a0ab "CPU_REPACK"
```
And we can see that when we call `initialize_tensors(ctx);` the repack set_tensor
is called:
```c++
        // randomize tensors
        initialize_tensors(ctx);
```
Skipping the first sentinel tensor:
```console
(gdb) p *tensor
$6 = {type = GGML_TYPE_Q4_0, buffer = 0x55555567c670, ne = {256, 16, 1, 1}, nb = {18, 144, 2304, 2304}, op = GGML_OP_NONE,
  op_params = {0 <repeats 16 times>}, flags = 0, src = {0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0}, view_src = 0x0,
  view_offs = 0, data = 0x55555569dcc0, name = "a", '\000' <repeats 62 times>,
  extra = 0x7ffff797b3a8 <ggml_repack_get_optimal_repack_type(ggml_tensor const*)::q4_0_8x8_q8_0>,
  padding = "\000\000\000\000\000\000\000"}

  1886	        auto OK = tensor_traits->repack(tensor, data, size);
(gdb)
repack: repack tensor a with q4_0_8x8
1887	        GGML_ASSERT(OK == 0);
```
So far so good then, we have repacked the input tensor a.

So then we have the callback user data setup and definition of the callback
lambda and we call `ggml_backend_compare_graph_backend`, and the first thing
to happen there is to that the graph is copied:
```c++
    struct ggml_backend_graph_copy copy = ggml_backend_graph_copy(backend2, graph);
```
```c++
struct ggml_backend_graph_copy ggml_backend_graph_copy(ggml_backend_t backend, struct ggml_cgraph * graph) {
    GGML_ASSERT(graph);
    struct ggml_hash_set hash_set = ggml_hash_set_new(graph->visited_hash_set.size);
    struct ggml_tensor ** node_copies = (ggml_tensor **) calloc(hash_set.size, sizeof(node_copies[0])); // NOLINT
    bool * node_init = (bool *) calloc(hash_set.size, sizeof(node_init[0]));

    struct ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead()*hash_set.size + ggml_graph_overhead_custom(graph->size, false),
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ true
    };

    struct ggml_context * ctx_allocated = ggml_init(params);
    struct ggml_context * ctx_unallocated = ggml_init(params);

    // dup nodes
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, node);
    }

    // allocate nodes
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx_allocated, backend);

    // copy data and init views
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        graph_copy_init_tensor(&hash_set, node_copies, node_init, node);
    }

    // build graph copy
    struct ggml_cgraph * graph_copy = ggml_new_graph_custom(ctx_allocated, graph->size, false);
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        struct ggml_tensor * node_copy = node_copies[ggml_hash_find(&hash_set, node)];
        graph_copy->nodes[i] = node_copy;
    }
    graph_copy->n_nodes = graph->n_nodes;

    ggml_hash_set_free(&hash_set);
    free(node_copies);
    free(node_init);

    return {
        /* .buffer           = */ buffer,
        /* .ctx_allocated    = */ ctx_allocated,
        /* .ctx_unallocated  = */ ctx_unallocated,
        /* .graph            = */ graph_copy,
    };
}
```
When we get to `graph_copy_init_tensor` and try to copy
```console
(gdb) p *node
$18 = {type = GGML_TYPE_F32, buffer = 0x55555567c670, ne = {16, 1, 1, 1}, nb = {4, 64, 64, 64}, op = GGML_OP_MUL_MAT, op_params = {
    0 <repeats 16 times>}, flags = 0, src = {0x555555691640, 0x555555691920, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0},
  view_src = 0x0, view_offs = 0, data = 0x5555556a09c0, name = "out", '\000' <repeats 60 times>, extra = 0x0,
  padding = "\000\000\000\000\000\000\000"}
```
```c++
static void graph_copy_init_tensor(struct ggml_hash_set * hash_set, struct ggml_tensor ** node_copies, bool * node_init, struct ggml_tensor * src) {
    ...
    else {
        ggml_backend_tensor_copy(src, dst);
    }
}
```
This will call into ggml-backend.cpp.
```c++
void ggml_backend_tensor_copy(struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    if (src == dst) {
        return;
    }

    if (ggml_backend_buffer_is_host(src->buffer)) {
        ggml_backend_tensor_set(dst, src->data, 0, ggml_nbytes(src));
    } else if (ggml_backend_buffer_is_host(dst->buffer)) {
        ggml_backend_tensor_get(src, dst->data, 0, ggml_nbytes(src));
```
```c++
void ggml_backend_tensor_get(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    if (size == 0) {
        return;
    }

    GGML_ASSERT(buf != NULL && "tensor buffer not set");
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor(buf, tensor, data, offset, size);
}
```
But `buf` is of repack extra buffer type:
```console
(gdb) p ggml_backend_buffer_name(buf)
$24 = 0x7ffff794a0ab "CPU_REPACK"
```
And it does not have a `get_tensor` function associated with it:
```
(gdb) p buf.iface.get_tensor
$26 = (void (*)(ggml_backend_buffer_t, const ggml_tensor *, void *, size_t, size_t)) 0x0
```
Would it perhaps make sense to add a `get_tensor` function to the repack buffer
type?  
I'm thinking if this is only used for copying tensors between backends and not
used for comparing the tensors data (my understanding is that only the
result/output tensors are compared) then it might be ok to just copy the
repackaged data.

Trying this out I get:
```console
[MUL_MAT] NaN at index 0 (CPU-alderlake=0.000000 CPU-ref=-nan)   
  MUL_MAT(type_a=q4_0,type_b=f32,m=16,n=1,k=256,bs=[1,1],nr=[1,1],per=[0,1,2,3],v=0,o=1): FAIL
```
So the copying of data happens before the graph computation in
`ggml_backend_compare_graph_backend`. There is only one graph and it is copied
into backend2 (the reference CPU). For non-repack this would just copy the input
tensors which is fine and what we actually want so that we have the same input
for both backends, it is the result we are interested in. But in the case of
repack, when we set the input tensors data repack's set_tensor will have repacked
the data and this is what is being copied (since we added a get_tensor for repack).
So the second backend will execute with repacked data and the first backend will
also operate on repacked data which is not what we want do to. We want to compare
the results of repack againt the reference implementation without repack.

What we should be testing:
* Backend1: Regular data with repack computation optimizations
* Backend2: Regular data with regular computation
* Compare the results

So how do we handle this?

_wip_

In the test tensor data is initialized using (for each tensor):
```c++
static void init_tensor_uniform(ggml_tensor * tensor, float min = -1.0f, float max = 1.0f) {
    size_t nels = ggml_nelements(tensor);
    std::vector<float> data(nels);
    // random values generated using threads...
    ...

    if (tensor->type == GGML_TYPE_F32 || tensor->type == GGML_TYPE_I32) {
        ggml_backend_tensor_set(tensor, data.data(), 0, nels * sizeof(float));
    } else if (ggml_is_quantized(tensor->type) || tensor->type == GGML_TYPE_F16 || tensor->type == GGML_TYPE_BF16) {
      ...

        ggml_backend_tensor_set(tensor, dataq.data(), 0, dataq.size());
    }
```
So at this point we have the un-repacked data in `dataq`. After this call if
the repack buffer type is used for the tensor, the repack set_tensor will be
called which will repack the data in the tensor's data buffer. Perhaps we could
store the original dataq somewhere, but the problem is that we need enable
the backend2 to use this for its computation was we want it to use the original
data.

Hmm, I wonder if there is a way to create a new ggml_backend_buffer_type for the
test, which in wraps the repack backend buffer type. And this specific type would
be able to store the original data, and it would implement get_tensor to return
this data instead of the repacked data. So perhaps we can add this as an additional
ggml_backend_cpu_get_extra_buffer_types in a similar way to 
```c++
#ifdef GGML_USE_CPU_REPACK
        if (ggml_backend_cpu_repack_buffer_type()) {
            bufts.push_back(ggml_backend_cpu_repack_buffer_type());
        }
#endif

#ifdef GGML_BUILD_TESTS
        if (ggml_backend_cpu_wrapper_repack_buffer_type()) {
            bufts.push_back(ggml_backend_cpu_wrapper_repack_buffer_type());
        }
#endif
```
And the we might be able to store the data that is passed to set_tensor in this
new buffer type and return it in get_tensor. This way we can use the repack
implementation for backend1 and the original data for backend2.

