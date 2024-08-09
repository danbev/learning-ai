## GGML Quantization
In broad terms this is about taking a floating point number like a single
precision floating point number (32 bits), or a half precision floating point
number (16 bits), and converting it to a fixed point number with a fixed number
of bits. 


### scale
```
         (x_max - x_min)         floating point range
scale =  ---------------         --------------------
         (q_max - q_min)           quantized range
           
```

### `ggml_half`
This typedef is defined in `ggml/src/ggml-common.h`:
```c
typedef uint16_t ggml_half;
```
So this is an unsigned 16 bit integer, that is 2 bytes. And unsigned meaning
that it can only store positive values.

### `ggml_half2`
This typedef is defined in `ggml/src/ggml-common.h`:
```c
typedef uint32_t ggml_half2;
```
And this is a 32 bit unsigned integer, that is 4 bytes, and like `ggml_half` it
can only store positive values.


* QI is the quantized value
* QK is the number of bits used for quantization
* QR is the ratio of the quantized value and the number for which it is a quantization(?)

### Blocks
In the coming sections we will look at types that are used in ggml and the all
start with block_ and it was not clear to me what this meant and why blocks are
used. Blocks are simply tensors that are divided into blocks of a certain size
and then quantized individually. As we will see we have a scaling factor when we
quantize which is calculated based on the maximum value in the block. If just
one or a few data points are extreme outliers (very high or very low compared to
the rest of the data), they can disproportionately influence the scale factor.
This is because the scale factor is often chosen to accommodate the maximum
absolute value in the tensor.
So instead the tensors are flattened into vectors and then divided into blocks
of a certain size like 32, 64, or 128 elements. Each block is the scaled
individually based on its own max absolute value. Outliers affect only the block
they are in, rather than the entire dataset. 

### `block_q4_0`
This struct is defined in `ggml/src/ggml-common.h`:
```c
#define QK4_0 32

typedef struct {
    ggml_half d;           // delta
    uint8_t qs[QK4_0 / 2]; // nibbles (4 bits) / quants
} block_q4_0;
```
The delta (`d`) is used to map the range of float values into the range of
integer values. This is calculated by finding the maximum value in the block of
float values and dividing it by the maximum value that can be represented by the
quantized value. In this case we have 4 bits which gives a range of 0 to 15
(0000-1111).
```
delta = max_value / quantized range
delta = max_value / 15                (1111b)
```

`qs` is where the quantized values are stored. So we have a array of 16 elements
(32/2=16), and notice the type is `uint8_t` which is 1 byte so each entry can
hold 8 bits.

Now each quantized value is 4 bits so we can store two in each entry in this
array:
```
   [ nibble0  ][ nibble1   ]
   +--+--+--+--+--+--+--+--+
0  |0 |1 |2 |3 |4 |5 |6 |7 |
   +--+--+--+--+--+--+--+--+
   ...
   +--+--+--+--+--+--+--+--+
15 |0 |1 |2 |3 |4 |5 |6 |7 |
   +--+--+--+--+--+--+--+--+
   [ nibble0  ][ nibble1   ]
```

#### Quantization
First we need to calculate the delta which is defined as:
```
org_values = {0.2, 0.3, 0.4, 0.5}

delta = max_value / 15
delta = 0.5 / 15
delta ~ 0.0333
```
This means that all values in my set of floating point numbers will be divided
by this delta so that they will be able to be represented by one of the number
in the range from 0-15. And the max value will be mapped to the value 15 because
0.0333 * 15 = 0.4995 ~ 0.5

The we quantize using the formula:
```
quantized_value = round(org_value / delta)

0.2 -> 0.2 / 0.0333 = 6
0.3 -> 0.3 / 0.0333 = 9
0.4 -> 0.4 / 0.0333 = 12
0.5 -> 0.5 / 0.0333 = 15

quantized_values = {6, 9, 12, 15}
```

#### Dequantization
To dequantize we use the formula:
```
org_value = quantized_value * delta

{6, 9, 12, 15}
6 -> 6 * 0.0333 = 0.1998
9 -> 9 * 0.0333 = 0.2997
12 -> 12 * 0.0333 = 0.3996
15 -> 15 * 0.0333 = 0.4995
```
So this is how the delta stored in the block struct is used.

Now, if we take a look at how quantization works in ggml this is done using
type traits. For example in `ggml/src/ggml.c` we have:
```c
static const ggml_type_traits_t type_traits[GGML_TYPE_COUNT] = {
    ...
    [GGML_TYPE_Q4_K] = {
        .type_name                = "q4_K",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_q4_K),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q4_K,
        .from_float               = quantize_row_q4_K,
        .from_float_reference     = (ggml_from_float_t) quantize_row_q4_K_reference,
        .vec_dot                  = ggml_vec_dot_q4_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    ...
};
```
There is an example in [ggml-quants.c](fundamental/ggml/src/ggml-quants.c) which
how this type trait can be accessed.
Lets take a look at `from_float` which is a function pointer to
`quantize_row_q4_K` and is defined in `ggml-quants.c`:
```c
void quantize_row_q4_0(const float * restrict x, void * restrict y, int64_t k) {
    quantize_row_q4_0_reference(x, y, k);
}

void quantize_row_q4_0_reference(const float * restrict x, block_q4_0 * restrict y, int64_t k) {
    static const int qk = QK4_0;
    printf("k = %ld\n", k);
    printf("qk = %d\n", qk);

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max
        float max  = 0.0f;

        for (int j = 0; j < qk; j++) {
            const float v = x[i*qk + j];
            if (amax < fabsf(v)) {
                amax = fabsf(v);
                max  = v;
            }
        }

        const float d  = max / -8;
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        for (int j = 0; j < qk/2; ++j) {
            const float x0 = x[i*qk + 0    + j]*id;
            const float x1 = x[i*qk + qk/2 + j]*id;

            const uint8_t xi0 = MIN(15, (int8_t)(x0 + 8.5f));
            const uint8_t xi1 = MIN(15, (int8_t)(x1 + 8.5f));

            y[i].qs[j]  = xi0;
            y[i].qs[j] |= xi1 << 4;
        }
    }
}
```
The first thing to notice is that the value of `k` which is the number of
elememnt int the x array or floats, which are the float values we want to
quantize and y is a pointer to the type trait. x has to be divisable by 32 which
is asserted. Following the number of blocks, `nb`, is calculated and then it
iterates over the first block.

For each block an absoute max (one that ignores the sign) and a max will be
stored. So it goes through the block and extracts the value from the block by
indexing into x using the block (i) and the block size, plus the current element
of the block we are at.


### `block_q4_1`
This is very similar to `block_q4_0` but we have an additional field in the
struct which is a union (so only one of the members can be used a one time):
```c
#define QK4_1 32

typedef struct {
    union {
        struct {
            ggml_half d;
            ggml_half m;
        } GGML_COMMON_AGGR;
        ggml_half2 dm;
    };
    uint8_t qs[QK4_1 / 2];
} block_q4_1;
```
So we can now have either a struct of a `ggml_half2` (the same information only
packed, two ggml_half's, into `ggml_half2` I think) member. The `m` member is
used to store the smallest value in the block of float values.

So the above is a common patterns where `_0` there is only the one delta value
but with `_1` there is an additional field to store data which in this case is
the minimum. I'm thinking that `_0` is because it needs at least the delta.

#### Quantization
Calculate min and delta using the forumla:
```
delta = (max_value - min_value) / 15
```
The 15 comes from the number of bits used for quantization which in this case
is 4 (1111b, 15d).

Then quantize using the formula:
```
quantized_value = round((org_value - min_value) / delta)

org_values = {0.2, 0.3, 0.4, 0.5}
min_value = 0.2
delta = (0.5 - 0.2) / 15
delta = 0.3 / 15
delta ~ 0.02
0.2 -> (0.2 - 0.2) / 0.02 = 0
0.3 -> (0.3 - 0.2) / 0.02 = 5
0.4 -> (0.4 - 0.2) / 0.02 = 10
0.5 -> (0.5 - 0.2) / 0.02 = 15
```

#### Dequantization
To dequantize we use the formula:
```
org_value = (quantized_value * delta) + min_value

quantized values = {0, 5, 10, 15}
0 -> (0 * 0.02) + 0.2 = 0.2
5 -> (5 * 0.02) + 0.2 = 0.3
10 -> (10 * 0.02) + 0.2 = 0.4
15 -> (15 * 0.02) + 0.2 = 0.5
```

### Naming Convention
So the above is a common patterns where `_0` there is only the one delta value
but with `_1` there is an additional field to store data which in this case is
the minimum. I'm thinking that `_0` is because it needs at least the delta which
is used to dequantize the quantized value when we scale the quantized value back
to the "original" (there is/maybe some loss).

### `block_q5_0`
We now have 5 bits (11111b, 32 values) to quantize the values:
```c
#define QK5_0 32

typedef struct {
    ggml_half d;           // delta
    uint8_t qh[4];         // 5-th bit of quants
    uint8_t qs[QK5_0 / 2]; // nibbles / quants
} block_q5_0;
```
Notice that `qs` is in fact the same size as for `block_q4_0` but we have an
additional field `qh` which has an array of 4 (0-4). This is used to store the
5th bit of the quantized value.

`qs` is where are quantized values are stored.
So we have a array of 16 elements and notice the type is `uint8_t` which is 1
byte so each entry can hold 8 bits. Now each quantized value is 4 bits so we can
store two in each entry in this array:
```
   +--+--+--+--+--+--+--+--+
0  |0 |1 |2 |3 |4 |5 |6 |7 |
   +--+--+--+--+--+--+--+--+
   ...
   +--+--+--+--+--+--+--+--+
15 |0 |1 |2 |3 |4 |5 |6 |7 |
   +--+--+--+--+--+--+--+--+
   [ nibble0  ][ nibble1   ]
```
With this we can store 32 quantized values in this array.
`qh` (h for high) is used to store the 5th bit of the quantized value so this is
16 bits in total, one for each entry in `qs` (the quantized values).

Like before we first need to calculate the delta:
```
delta = max_value / 31
```
This time we use 31 because we have 5 bits (11111b, 31d).
The to get the quantized values we use the formula:
```
quantized_value = round(org_value / delta)
```
For example:
```
org_values = {0.2, 0.3, 0.4, 0.5}
max_value = 0.5
delta = 0.5 / 31 = 0.0161

0.2 -> 0.2 / 0.0161 = 12
0.3 -> 0.3 / 0.0161 = 18
0.4 -> 0.4 / 0.0161 = 25
0.5 -> 0.5 / 0.0161 = 31

12 (01100b) = qs[0] = 1100b  gh[0] = 0
18 (10010b) = qs[1] = 0010b  gh[1] = 1
25 (11001b) = qs[2] = 1001b  gh[2] = 1
31 (11111b) = qs[3] = 1111b  gh[3] = 1
```

### `block_q5_1`
This struct is defined as follows:
```c
#define QK5_1 32

typedef struct {
    union {
        struct {
            ggml_half d; // delta
            ggml_half m; // min
        } GGML_COMMON_AGGR;
        ggml_half2 dm;
    };
    uint8_t qh[4];         // 5-th bit of quants
    uint8_t qs[QK5_1 / 2]; // nibbles / quants
} block_q5_1;
``
So this is very similar to `block_q5_0` and similar in the same way as
`block_q4_1` is to `block_q4_0`.

### `block_q8_0`
```c
#define QK8_0 32
typedef struct {
    ggml_half d;       // delta
    int8_t  qs[QK8_0]; // quants
} block_q8_0;
```
This is pretty similar to what we have seen before but notice that the quantized
values array are now 32 elements long. And the type is `int8_t` which is 1 byte
and not `uint8_t`. This is because the lower quantization blocks we have seen so
far is because the values are all positive, like for 4 bits we have 0000b-1111b
(0-15d). But now we have 8 bits so we can represent negative values as well
which as another advantage of having symmetric quantization.
But other than that the quantization is the same as before.


### `block_q8_1`
```c
typedef struct {
    union {
        struct {
            ggml_half d; // delta
            ggml_half s; // d * sum(qs[i])
        } GGML_COMMON_AGGR;
        ggml_half2 ds;
    };
    int8_t qs[QK8_1]; // quants
} block_q8_1;
```
We have `d` which is the delta/scale factor, but not instead of a minimum value
we have `s` which is the sum of the quantized values.

#### Quantization
```
data={0.1,0.2,0.3,…,3.2}

delta = (max(data) - min(data)) / 255
delta = (3.2 - 0.1) / 255 = 0.012549
delta = 0.012549

quantized_value = round((org_value - min_value) / delta)

0.1 -> (0.1 - 0.1) / 0.012549 = 0   -> qs[0] = 0
0.2 -> (0.2 - 0.1) / 0.012549 = 8   -> qs[1] = 8
0.3 -> (0.3 - 0.1) / 0.012549 = 16  -> qs[2] = 16
...
3.2 -> (3.2 - 0.1) / 0.012549 = 255 -> qs[255] = 255

sum_qs = 0 + 8 + 16 + ... + 255
s = d * sum_qs
```

#### Dequantization
```
org_value = quantized_value * d

0 -> 0 * 0.012549 = 0.0
8 -> 8 * 0.012549 = 0.100392
16 -> 16 * 0.012549 = 0.200784
...
255 -> 255 * 0.012549 = 3.199995
```

### `block_q2_K`
This struct is defined as follows:
```c
#define QK_K 256

//
// Super-block quantization structures
//

// 2-bit quantization
// weight is represented as x = a * q + b
// 16 blocks of 16 elements each
// Effectively 2.625 bits per weight
typedef struct {
    uint8_t scales[QK_K/16]; // scales and mins, quantized with 4 bits
    uint8_t qs[QK_K/4];      // quants
    union {
        struct {
            ggml_half d;    // super-block scale for quantized scales
            ggml_half dmin; // super-block scale for quantized mins
        } GGML_COMMON_AGGR;
        ggml_half2 dm;
    };
} block_q2_K;
```
So x is the weight, a is the scale factor, and b is the minimum value or offset.
Since we are only using 2 bits we only have 00, 01, 10, 11 quantization levels.



__wip__

```console
$ gdb --args bin/quants
(gdb) ptype  q4_0
type = struct {
    ggml_half d;
    uint8_t qs[16];
} *
(gdb) p sizeof(uint8_t)
$2 = 1
(gdb) ptype ggml_half
type = unsigned short
(gdb) p sizeof(ggml_half)
$3 = 2
```
And in this case `ggml_half` is 2 bytes so 16 bits.


### restrict
When a pointer is declared with restrict, the compiler can make certain
assumptions and perform optimizations that would otherwise be unsafe due to
potential aliasing. For example:
* The compiler can keep the value pointed to in a register instead of reloading
it from memory on each access, since no other pointer can modify that memory
location.
* The compiler can reorder read/write operations involving the restrict pointer,
as there is no risk of another pointer accessing the same memory concurrently.
* The compiler can perform vectorization and other loop optimizations more
aggressively, as there is no risk of data dependencies between iterations due to
 aliasing.

### Bits Per Weight (BPW)
