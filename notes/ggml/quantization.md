## GGML Quantization
In broad terms this is about taking a floating point number like a single
precision floating point number (32 bits), or a half precision floating point
number (16 bits), and converting it to a fixed point number with a fixed number
of bits. We do this to save space and memory and also to speed up computations.

The type of quantization to use is not just a matter of the level of precision
that we want to maintain which I thought initially, but the actual data that
we are quantizing also matters and should influence the choice of quantization
strategy/type. I'm just mentioning this as it was not obivious to me at first,
and I'll include examples that illustrate this.

### Symmetric quantization
This is where we map zero in the original data range to zero in the quantized
representation.

### Asymmetric quantization
This type of quantization allows for a non-zero offset in the quantized, so the
zero point can be shifted making it asymmetric with respect to zero in the
original data range. This is useful when the data has a non-zero mean or
when the distribution of values is not centered around zero.

### scale
So in the following we have a range of float values represented by by `r_min`
and `r_max`, and a range of quantized values represented by `q_min` and `q_max`.
```
         (r_max - r_min)         floating point range
scale =  ---------------         --------------------
         (q_max - q_min)           quantized range
```
Notice that the scale is just the ratio of the distance/range of the floating
point values and the quantized values. I might help to view these as points in
a coordinate system.
```
Float Values (r)
     ↑
 2.4 |────────────────●  ← Point 2: (q_max, r_max) = (15, 2.4)
     │               ╱
 1.2 │            ╱
     │         ╱
 0.0 │      ╱
     │   ╱
-1.2 ●╱──────────────────→ Quantized Values (q)
     0     5    10    15
     ↑
Point 1: (q_min, r_min) = (0, -1.2)

So we have two points:
(0, -1.2)
(15, 2.4)

        (r_max - r_min) = (2.4 - (-1.2))   3.6
scale = ------------------------------- = ------ = 0.24
        (q_max - q_min) = (15  -  0         15
```
Notice that this is just like caculating the slope (y2 - y1) / (x2 - x1). So
this tells us that for every 1 unit increase in the quantized values we have
0.24 unit increase in the floating point values.
```
quantized 0  → float -1.2
quantized 1  → float -0.96
quantized 2  → float -0.72
quantized 3  → float -0.48
quantized 4  → float -0.24
quantized 5  → float 0.0
quantized 6  → float 0.24
quantized 7  → float 0.48
quantized 8  → float 0.72
quantized 9  → float 0.96
quantized 10 → float 1.2
quantized 11 → float 1.44
quantized 12 → float 1.68
quantized 13 → float 1.92
quantized 14 → float 2.16
quantized 15 → float 2.4
```

To quantize a float, a value on the y axis, we calculate the distance from the
minimum value on the y axis, `r_min`, which is how far up the y axis am I from
the starting point. And then we want to figure out, so if each x-step moves me
0.24 unit up the y axis, how many x-steps do I need to take to travel the
distance calculated. For this we use the scale:
```
quantized_value = round((float_value - r_min) / scale)

Example float -1.2:
quantized_value = round((-1.2 - (-1.2)) / 0.24)
                = round((0) / 0.24)
                = round(0) = 0

Example float -0.96:
quantized_value = round((-0.96 - (-1.2)) / 0.24)
                = round((0.24) / 0.24)
                = round(1) = 1

Example float 0.0:
quantized_value = round((0.0 - (-1.2)) / 0.24)
                = round((1.2) / 0.24)
                = round(5) = 5
```
So we are taking the float value, the y axis and subtracting the minium

To dequantize we do:
```
float_value = scale * quantized_value + offset
offset      = r_min - (scale * q_min)

offset = -1.2 - (0.24 * 0) = -1.2

Example:
Quantized Value: 0
float_value = 0.24 * 0 + (-1.2) = -1.2

Quantized Value: 1
float_value = 0.24 * 1 + (-1.2) = -0.96
```
Notice that this is basically `y = mx + b`:
```
float_value = scale × quantized_value + offset
     ↑           ↑         ↑             ↑
   y-axis     slope      x-axis      y-intercept
```
And this is called linear quantization. The offset is the zero point which is

### zero point
```
zero_point = round(q_min - (r_min / scale))
```

### Notation
In this document and in ggml there is the following names/notation:

* QI is the quantized value
* QK is the number of bits used for quantization
* QR is the ratio of the quantized value and the number for which it is a quantization(?)

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

### Blocks
In the coming sections we will look at types that are used in ggml and the all
start with `block_` and it was not clear to me what this meant and why blocks
are used. Blocks are simply tensors that are divided into blocks of a certain
size and then quantized individually. As we will see we have a scaling factor
when we quantize which is calculated based on the maximum value in the block. If
just one or a few data points are extreme outliers (very high or very low
compared to the rest of the data), they can disproportionately influence the
scale factor. This is because the scale factor is often chosen to accommodate
the maximum absolute value in the tensor.
So instead the tensors are flattened into vectors and then divided into blocks
of a certain size like 32, 64, or 128 elements. Each block is then scaled
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
integer values. 
`qs` is where the quantized values are stored. So we have a array of 16 elements
(32/2=16), and notice the type is `uint8_t` which is 1 byte so each entry can
hold 8 bits.

Now each quantized value is 4 bits so we can store two in each entry in this
array:
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

#### Quantization
To quantize the following function is called:
```c++
void quantize_row_q4_0_ref(const float * GGML_RESTRICT x, block_q4_0 * GGML_RESTRICT y, int64_t k) {
    static const int qk = QK4_0;

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
So this function takes in a pointer to float `x`, a pointer to `block_q4_0` which
is `y`, and `k` which is the size of x. This has be be divisable by 32.
We calculate the number of blocks `nb` and then iterate over them. For each
block we calculate the absolute max and the max value that block (32 value).

After the first loop we use max value to compute the delta/scale value:
```c++
        const float d  = max / -8;
        const float id = d ? 1.0f/d : 0.0f;
```
So each float32 (single precision) value is going to be represented with 4 bits.
The `block_q4_0` stores its members as `uint8_t` which is 1 byte (8 bits) and
unsigned so it can only store positive values. That gives as a range of [0-15]
(0000b-1111b) which is 16 values. 

But notice that the code above is using -8 as the denominator and this is creating
a range of [-8, 7]. Why do this? With this range 0 is always maps to the center
of our quantization range and using [0-15] range might not, it depends on the
actual values.

Also notice that this is using the inverse delta `id` which is done so that
division can be avoided and instead just a multiplication operation performed
instead. The "real" delta/scale is the only thing stored in the block as that
is what is required for dequantization.

And notice that in the quantization stage later we are adding 8.5 to the
quantized value:
```c++
            const uint8_t xi0 = MIN(15, (int8_t)(x0 + 8.5f));
            const uint8_t xi1 = MIN(15, (int8_t)(x1 + 8.5f));
```
This for rounding and storing the quantized value in the range of [0-15]. And
we will see later that we subtract 8 when we dequantize the value to get it back
into the range of [-8, 7].

```
Input: [2.1, 2.3, 2.5, 2.7, 2.9]

max = 2.9  // extremal value
d = 2.9 / -8 = -0.3625
id = 1.0 / -0.3625 = -2.758

// Quantize each value:
2.1 * -2.758 = -5.792 → -5.792 + 8.5 = 2.708 → stored as 2
2.3 * -2.758 = -6.343 → -6.343 + 8.5 = 2.157 → stored as 2  
2.5 * -2.758 = -6.895 → -6.895 + 8.5 = 1.605 → stored as 1
2.7 * -2.758 = -7.447 → -7.447 + 8.5 = 1.053 → stored as 1
2.9 * -2.758 = -7.998 → -7.998 + 8.5 = 0.502 → stored as 0
```
Notice that there are multiple values that map to the same quantized value!

#### Dequantization
```
4-bit signed integers (two's complement):
Binary   Decimal
1000  →    -8
1001  →    -7
1010  →    -6
1011  →    -5
1100  →    -4
1101  →    -3
1110  →    -2
1111  →    -1
0000  →     0
0001  →    +1
0010  →    +2
0011  →    +3
0100  →    +4
0101  →    +5
0110  →    +6
0111  →    +7
```
So we have 16 different values that we can represent with 4 bits.
```
Input: [1.0, -0.5, 3.2, 0.8]
max = 3.2

d = 3.2 / -8 = -0.4
id = 1.0 / d = 1.0 / -0.4 = -2.5

3.2 * id = 3.2 * -2.5 = -8.0   (quantized value)

Add offset of 8.5:
-8.0 + 8.5 = 0.5 → rounds to 0
```
```
Input: [1.0, -0.5, -6.4, 0.8]
max = -6.4 

d = max / -8 = -6.4 / -8 = 0.8
id = 1.0 / d = 1.0 / 0.8 = 1.25

-6.4 * id = -6.4 * 1.25 = -8.0  (quantized value)

Add offset of 8.5:
-8.0 + 8.5 = 0.5 → rounds to 0
```
The addition of 8.5 is to bring the values into the range of [0.5-15.5] so that
we can store them in the `qs` array which is an array of 16 elements of type
uint8_t (notice that this is unsigned). Casting will truncate the decimal part
so the 0.5 ensure rounding.

Then when we dequantize we need to "move" back into the original range, so
from [0-15] to [-8-7], so we have to subtract by 8 (the 0.5 was only for rounding
and not required at this stage). So we have values in the range of [0-15] in
"storage" and then to get the quantized values we have to subtract 8:
```c++
void dequantize_row_q4_0(const block_q4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    static const int qk = QK4_0;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);

        for (int j = 0; j < qk/2; ++j) {
            // First we extract the lower bits:
            const int x0 = (x[i].qs[j] & 0x0F) - 8;
            // And then the upper bits:
            const int x1 = (x[i].qs[j] >>   4) - 8;

            y[i*qk + j + 0   ] = x0*d; // scale back to float32 value
            y[i*qk + j + qk/2] = x1*d; // scale back to float32 value
        }
    }
}
```


### ggml quantization type traits

Now, if we take a look at how quantization works in ggml this is done using
type traits. For example in `ggml/src/ggml.c` we have:
```c
static const ggml_type_traits_t type_traits[GGML_TYPE_COUNT] = {
    ...
    [GGML_TYPE_Q4_0] = {
        .type_name                = "q4_0",
        .blck_size                = QK4_0,
        .type_size                = sizeof(block_q4_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q4_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q4_0_ref,
    },
    ...
};
```
There is an example in [ggml-quants.c](../fundamentals/ggml/src/ggml-quants.c) which
how this type trait can be accessed.

Lets take a look at `from_float` which is a function pointer to
`quantize_row_q4_K` and is defined in `ggml-quants.c`:
```c
// reference implementation for deterministic creation of model files
void quantize_row_q4_0_ref(const float * GGML_RESTRICT x, block_q4_0 * GGML_RESTRICT y, int64_t k) {
    static const int qk = QK4_0;

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
elements in the x array of floats, which are the float values we want to
quantize, and y is a pointer to the type trait. x has to be divisable by 32 which
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
packed, two `ggml_half`'s, into one `ggml_half2` I think) member. The `m` member is
used to store the smallest value (min) in the block of float values.

So each `block_q4_1` will be storing:
```
d  = uint8_t     = 16 bits (2 bytes) (delta/scale value)
m  = uint8_t     = 16 bits (2 bytes) (minimum value in the block)
qs = uint8_t[16] = 16 * 8 = 128 bits (16 bytes) (quantized values)

Total: 16 + 16 + 128 = 160 bits = 20 bytes
```

So the above is a common patterns where `_0` there is only the one delta value
but with `_1` there is an additional field to store data which in this case is
the minimum. I'm thinking that `_0` is because it needs at least the delta.

#### Quantization
```
min = 2.1, max = 2.9
d = (2.9 - 2.1) / 15 = 0.8 / 15 = 0.0533
id = 1.0 / 0.0533 = 18.75

// Quantize each value:
(2.1 - 2.1) * 18.75 = 0.0   → 0.0 + 0.5 = 0.5     → stored as 0
(2.3 - 2.1) * 18.75 = 3.75  → 3.75 + 0.5 = 4.25   → stored as 4
(2.5 - 2.1) * 18.75 = 7.5   → 7.5 + 0.5 = 8.0     → stored as 8
(2.7 - 2.1) * 18.75 = 11.25 → 11.25 + 0.5 = 11.75 → stored as 11
(2.9 - 2.1) * 18.75 = 15.0  → 15.0 + 0.5 = 15.5   → stored as 15
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

org_values             = {0.2, 0.3, 0.4, 0.5}
quantized->dequantized = {0.2, 0.3, 0.4, 0.5}
```

### `block_q4_K`
So lets start with what this is all about. This is about efficiently storing
multiple blocks.

Lets say we wanted to store 8 separete `block_q4_1` blocks:
```
One block_q4_1:
d  = uint8_t     = 16 bits (2 bytes) (delta/scale value)
m  = uint8_t     = 16 bits (2 bytes) (minimum value in the block)
qs = uint8_t[16] = 16 * 8 = 128 bits (16 bytes) (quantized values)

Total: 16 + 16 + 128 = 160 bits = 20 bytes

8 block_q4_1:
Total: 8 x 20 = 160 bytes
```

Now, a `block_q4_K` looks like this:
```c++
#define QK_K 256
#define K_SCALE_SIZE 12

// 4-bit quantization
// 8 blocks of 32 elements each
// weight is represented as x = a * q + b
// Effectively 4.5 bits per weight
typedef struct {
    GGML_EXTENSION union {
        struct {
            ggml_half d;    // super-block scale for quantized scales
            ggml_half dmin; // super-block scale for quantized mins
        } GGML_COMMON_AGGR_S;
        ggml_half2 dm;
    } GGML_COMMON_AGGR_U;
    uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uint8_t qs[QK_K/2];           // 4--bit quants
} block_q4_K;
```
And before we go into the details of this struct lets first look at what the
storage requirements for such a block in comparision with the 8 single
`block_q4_1` blocks:
```
d          = ggml_half = 16 bits (2 bytes) (super-scale for scales)
dmin       = ggml_half = 16 bits (2 bytes) (super-scale for mins)
scales[12] = 96 bits (12 bytes) (8 scales + 8 mins, 6 bits each)
qs[128]    = 1024 bits (128 bytes) (256 values, 4 bits each)

Total: 16 + 16 + 96 + 1024 = 1152 bits = 144 bytes
```
So notice that our 8 individual `block_q4_1` blocks would take 160 bytes, and
this single `block_q4_K` block takes 144 bytes, so we save 16 bytes which is
about 10% of the size. This is a significant saving when we have many blocks.

So how is this saving achieved?  
The "secret" is how the the instead of using full precision values for the
delta/scale and the minimum value, the K block quantizes these values. So there
is like two levels of quantization being used here.
```
8 block_q4_1:
8 full precision deltas/scales: 8 x 16 = 128 bits
8 full precision mins         : 8 x 16 = 128 bits
Parameter totals              : 256 bits = 32 bytes

block_q4_K:
2 super-scales/deltas         : 2 × 16 = 32 bits
8 quantized scales/deltas     : 8 × 6 = 48 bits  
8 quantized mins              : 8 × 6 = 48 bits
Parameter overhead            : 128 bits total
```
Notice that only 6 bits are availble for the scale and min values for each of
the 8 blocks. In the case of individual `block_q4_1` blocks we have 16 bits for
the scale and 16 bits for the minimum value which gives us plenty of precision.

So we have these precision constraints for the scales and min values in this
case which is why the code it not as straightforward as the standalone blocks.
So we have to find scale/min values that work well for quantization but also that
themselves can be quantized to 6 bits.

So we have a set of floating point values which we can imagine as points on the
y axis like we did previously, and on the x axis we have the quantized units.

What we want to do is to find a line that fits these points well and we can
use linear regression/least squares to find the best fit line. The slope of this
line is the scale, and the y-intercept is the minimum value. But we also have
additional constraits that the slope/intercept must fit in 6 bits.
```
float values[4]     = {10.0, 11.0, 13.0, 14.0};
2 bits quantization = [ 0 1 2 3 ] ("units" of quantization/level)

  ^
14-            *
  |
13-        *
  |
12-
  |
11-   *
  |
  |
10*
  |
  |
  |
  |
  |----|---|---|---|------->
  0    1   2   3   4

// Data points (level, float_value):
(0, 10.0)
(1, 11.0)
(2, 13.0)
(3, 14.0)
```
Now the line is just a mental model, or something that we could do manually
perhaps is a better way of thinking about it. But to actually perform this
operation we use calculus and derivatives to find the best fit line.

For any line `y = scale * x + min` we can calculate the error using:
```console
error_i = actual_value_i - predicted_value_i
error_i =            y_i - (scale × x_i + min)
```
For example, lets take point 1:
```console
point: (1, 11.0)
scale: 1.4
min: 9.9

predicted_value = (1.4   × 1 + 9.9) = 11.3
                  (scale × x + min)
error           = 11.0 - 11.3 = -0.3
```
So that is for a single value, but we want to calculate the total error for all
the points and also square them, so we can use the following formula:
```console
Total_Error = (f1_error)² + (f2_error)² + (f3_error)² + (f4_error)²

Total_Error = Σ(squared_error_i)
            = Σ(y_i - (scale × x_i + min))²
            =   (y₁ - (scale ×  x₁ + min))² +
                (y₂ - (scale ×  x₂ + min))² +
                (y₃ - (scale ×  x₃ + min))² +
                (y₄ - (scale ×  x₄ + min))²

Total_Error = (10.0 - (scale × 0 + min))² +
              (11.0 - (scale × 1 + min))² +
              (13.0 - (scale × 2 + min))² +
              (14.0 - (scale × 3 + min))²
```
Now, notice that we have two unknowns here, the `scale` and the `min`.
```
total_error = ∑(y_i - (scale * x_i + min))²
```
So we take the derivative with respect to `scale`:
```console
d(Total_Error)/d(scale) = d/d(scale) [Σ(y_i - (scale × x_i + min))²]
```
We move the derivative (d/d(scale)) inside the summation
(linearity of derivatives):
```console
d(Total_Error)/d(scale) = Σ(d/d(scale) [(y_i - (scale × x_i + min))²])
```
So we now have:
```
Σ( d/d(scale) [(y_i - (scale * x_i + min))²])
And just recall that d/d(scale) is a function so we can think of it as:
Σ( d/d(scale)([(y_i - (scale * x_i + min))²]))

So d/d(scale) is the derivative operator/function that takes
(y_i - scale * x_i + min))² as its input.

This can be helpful when we want to apply the chain rule:
Σ( d/d(scale)([(y_i - (scale * x_i + min))²]))
   {         outer function                 }
              { inner function             }
We start by taking the derivative of the outer function:

Let u = y_i - (scale * x_i + min)
So the outer function is u²:
d/du(u²) = 2u = 2(y_i - (scale * x_i + min))

And we also have to take the derivative of the inner function:
du/d(scale) = d/d(scale)(y_i - (scale * x_i + min))
            = - x_i

And then we combine the two:
2(y_i - (scale * x_i + min)) * (-x_i)
-2x_i(y_i - (scale * x_i + min))

And we can then place this back into the summation:

Σ( d/d(scale)([(y_i - (scale × x_i + min))²])) = ∑(-2x_i(y_i - (scale * x_i + min)))
                                               = -2Σ(x_i * (y_i - (scale * x_i + min)))
Set to zero:
-2Σ(x_i × (y_i - (scale × x_i + min))) = 0
Σ(x_i × (y_i - (scale × x_i + min))) = 0
Expand:
Σ(x_i × y_i - x_i × (scale × x_i + min)) = 0
Σ(x_i × y_i - x_i × scale × x_i - x_i × min) = 0
Σ(x_i × y_i - scale × x_i² - min × x_i) = 0

Σ(x_i × y_i) - Σ(scale × x_i²) - Σ(min × x_i) = 0

Σ(x_i × y_i) - scale × Σ(x_i²) - min × Σ(x_i) = 0

Σ(x_i × y_i) = scale × Σ(x_i²) + min × Σ(x_i)
```

And then we do something similar but for the `min`:
```console
d(Total_Error)/d(min) = d/d(min) [Σ(y_i - (scale × x_i + min))²]

Move the derivate inside of the summation:
= Σ [d/d(min) (y_i - (scale × x_i + min))²]

Apply the chain rule:
Let u = y_i - (scale × x_i + min), so we have u²

Chain rule: d(u²)/d(min) = 2u × du/d(min)

Find the derivative of the outer function:
d/du(u²) = 2u = 2(y_i - (scale × x_i + min))

Find the derivative of the inner function:
du/d(min) = d/d(min)[y_i - (scale × x_i + min)]
          = -1

Combine the two:
d/d(min)[(y_i - (scale × x_i + min))²] = 2(y_i - (scale × x_i + min)) × (-1)
                                       = -2(y_i - (scale × x_i + min))

And place this back into the summation:
Σ[d/d(min) (y_i - (scale × x_i + min))²] = Σ[-2(y_i - (scale × x_i + min))]
                                         = -2Σ(y_i - (scale × x_i + min))

Set to zero:
-2Σ(y_i - (scale × x_i + min)) = 0
Σ(y_i - (scale × x_i + min)) = 0
Σ(y_i) - scale×Σ(x_i) - min×n = 0

Rearranging:
Σ(y_i) = scale×Σ(x_i) + min×n
```
So that gives us two equations:
```
Equation 1: Σ(x_i × y_i) = scale × Σ(x_i²) + min × Σ(x_i)
Equation 2: Σ(y_i)       = scale × Σ(x_i)  + min × n
```
So we start by solving for `min`:
```console
Σ(y) = scale × Σ(x) + min × n

Subtract scale × Σ(x) from both sides:
Σ(y) - scale × Σ(x) = min × n

Divide by n:
min = [Σ(y) - scale × Σ(x)] / n
```
And then we can substitute this into the first equation:
```console
Σ(xy) = scale × Σ(x²) + min × Σ(x)

Σ(xy) = scale × Σ(x²) + [Σ(y) - scale × Σ(x)] / n × Σ(x)

n × Σ(xy) = n × scale × Σ(x²) + [Σ(y) - scale × Σ(x)] × Σ(x)
n × Σ(xy) = n × scale × Σ(x²) + Σ(y) × Σ(x) - scale × Σ(x) × Σ(x)
n × Σ(xy) = n × scale × Σ(x²) + Σ(y) × Σ(x) - scale × [Σ(x)]²

n × Σ(xy) - Σ(y) × Σ(x) = n × scale × Σ(x²) - scale × [Σ(x)]²
n × Σ(xy) - Σ(y) × Σ(x) = scale × [n × Σ(x²) - [Σ(x)]²]

scale = [n × Σ(xy) - Σ(y) × Σ(x)] / [n × Σ(x²) - [Σ(x)]²]

We can try this out with our example:
min = [Σ(y) - scale × Σ(x)] / n

n = 4, Σ(x) = 6, Σ(y) = 48, Σ(xy) = 79, Σ(x²) = 14

scale = [4×79 - 48×6] / [4×14 - 6²] = [316 - 288] / [56 - 36] = 28/20 = 1.4

min = [48 - 1.4×6] / 4 = [48 - 8.4] / 4 = 39.6/4 = 9.9
```
Notice that we need a number of values to be able to calculate the scale and
the min which is something we will see in the code later.

Now, with the scale and the min values calculated we have found the optimal
linear mapping from our input floating point values to our quantized units.
We can use this to quantized values:
```console
quantized_value = (float_value - min) / scale

10.0 → (10.0 - 9.9) / 1.4 = 0.1 / 1.4 ≈ 0
11.0 → (11.0 - 9.9) / 1.4 = 1.1 / 1.4 ≈ 1
13.0 → (13.0 - 9.9) / 1.4 = 3.1 / 1.4 ≈ 2
14.0 → (14.0 - 9.9) / 1.4 = 4.1 / 1.4 ≈ 3
```
And we can dequantize the values using the inverse mapping:
```console
reconstructed_value = scale × quantized_value + min

0 → 1.4 × 0 + 9.9 = 9.9  ≈ 10.0
1 → 1.4 × 1 + 9.9 = 11.3 ≈ 11.0
2 → 1.4 × 2 + 9.9 = 12.7 ≈ 13.0
3 → 1.4 × 3 + 9.9 = 14.1 ≈ 14.0
```

```
float values[4]     = {10.0, 11.0, 13.0, 14.0};
2 bits quantization = [ 0 1 2 3 ] ("units" of quantization/level)

    +--------------------+
    | x  | y  | xy | x^2 |
    +--------------------+
    | 0  | 10 |  0 | 0   |
    +----+----+----+-----+
    | 1  | 11 | 11 | 1   |
    +----+----+----+-----+
    | 2  | 13 | 26 | 4   |
    +----+----+----+-----+
    | 3  | 14 | 42 | 9   |
    +----+----+----+-----+

Sums
    +----+----+----+-----+
    | 6  | 48 | 79 | 14  |
    +----+----+----+-----+

y = mx + b

n = number of floating point values = 4

m = slope (scale) = (n * Σxy - Σx * Σy) / (n * Σx^2 - (Σx)^2)
  = (4 * 79 - 6 * 48) / (4 * 14 - 6^2)
  = (316 - 288) / (56 - 36)
  = 28 / 20
  = 1.4

So this gives us a slope of 1.4 which is the scale value.

b = y-intercept (minimum) = (Σy - m * Σx) / n
  = (48 - 1.4 * 6) / 4
  = (48 - 8.4) / 4
  = 39.6 / 4
  = 9.9

So that gives us:
y = 1.4x + 9.9

x = 0:
y = 1.4 * 0 + 9.9 = 9.9     (error: 0.1)

x = 1:
y = 1.4 * 1 + 9.9 = 11.3    (error: 0.3)

x = 2:
y = 1.4 * 2 + 9.9 = 12.7    (error: 0.3)

x = 3:
y = 1.4 * 3 + 9.9 = 14.1    (error: 0.1)

Total squared error: 
(0.1^2 + 0.3^2 + 0.3^2 + 0.1^2) = 0.20
```

So that sound great, and lets take a look at how this works in practice.

So lets say we have 1 block of 256 elements, so we have 8 blocks of 32 elements.
```c++
void quantize_row_q4_K_ref(const float * GGML_RESTRICT x, block_q4_K * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;
```
```console
(gdb) p k
$7 = 256

(gdb) p nb
$8 = 1
```

```c++
    uint8_t L[QK_K];
    uint8_t Laux[32];
    float   weights[32];
    float mins[QK_K/32];
    float scales[QK_K/32];
```
Lets take a look at the sizes of these arrays:
```console
(gdb) p sizeof(L)
$13 = 256
(gdb) p sizeof(Laux)
$21 = 32
(gdb) p sizeof(weights)/sizeof(float)
$22 = 32
(gdb) p sizeof(mins)/sizeof(float)
$18 = 8
(gdb) p sizeof(scales)/sizeof(float)
$19 = 8
```

So the following loop will iteratate over the number of blocks, in this case
1 block:
```c++
    for (int i = 0; i < nb; i++) {
        float max_scale = 0;
        float max_min = 0;

        // This will iterate over the 8 blocks of 32 elements each.
        for (int j = 0; j < QK_K/32; ++j) {

            float sum_x2 = 0;
            for (int l = 0; l < 32; ++l) {
                sum_x2 += x[32*j + l] * x[32*j + l];
            }
```
So at this we ahve the sum of the `x^2` for the 32 elements in the block for
table above and ∑(x^2) in the equation:
```console
                                                ↓
m = slope (scale) = (n * Σxy - Σx * Σy) / (n * Σx^2 - (Σx)^2)
```

The next part of the loop is something that we did not see in the above
discussion as this is specific to the quantization in ggml. This is dealing with
the contstraint that the scale and min values must fit in 6 bits:
```c++
            float av_x = sqrtf(sum_x2/32);
            for (int l = 0; l < 32; ++l) {
                weights[l] = av_x + fabsf(x[32*j + l]);
            }
```
First we have a root mean square of the 32 values which gives us a "typical"
magnitude which is stored in `av_x` (average x?). The for loop then calculates
a weight for each of the 32 values which is average magnitude plus the absolute
value of the current value. So larger values will have a larger weight and
smaller values will have a smaller weight.

Next we are going to find the optimal scale and min values for quantizing this
block of 32 values, with the contraints that values must fit into limited range.
```c++
            scales[j] = make_qkx2_quants(32, 15, x + 32*j, weights, L + 32*j, &mins[j], Laux, -1.f, 0.1f, 20, false);
```
The first argument is the number of elements of the block which is 32, and after
that we have the number of bits that we want to quantize the values to, which is
15 bits. The third argument is the pointer to the current block. Following
that we have the weights we just calculated. Next we have the pointer to the
output quantized values, and then then a pointer to the output min values. Next
is Laux which is a temporary array that I think is used to store and compare
different quantization results. Then we have a few "search" parameters which
I'll try to address in the code/notes below.

```c++
static float make_qkx2_quants(int n, int nmax, const float * GGML_RESTRICT x, const float * GGML_RESTRICT weights,
        uint8_t * GGML_RESTRICT L, float * GGML_RESTRICT the_min, uint8_t * GGML_RESTRICT Laux,
        float rmin, float rdelta, int nstep, bool use_mad) {
    float min = x[0];
    float max = x[0];
    float sum_w = weights[0];
    float sum_x = sum_w * x[0];
```
This is setting the initial min, and max value to the first value in the block
which is:
```console
(gdb) p x[0]
$8 = 10
(gdb) p weights[0]
$9 = 14.2793102
```
And we also set the intial sum of the weights and the sum of the weighted
values to the first value in the block multiplied by the weight of that value:
```console
(gdb) p sum_x
$10 = 142.793106
```

Next we set the rest of the min, max, and sum values:
```c++
    for (int i = 1; i < n; ++i) {
        if (x[i] < min) min = x[i];
        if (x[i] > max) max = x[i];
        float w = weights[i];
        sum_w += w;
        sum_x += w * x[i];
    }
```
We want 0 to be the minimum value, so we adjust the min to zero if it is g
reater than zero, and if the max is equal to the min we set all the 
quantized values to zero and return 0 as the scale:
```c++
    if (min > 0) min = 0;
    if (max == min) {
        for (int i = 0; i < n; ++i) L[i] = 0;
        *the_min = -min;
        return 0.f;
    }
```
The following will iterate over all then values in the current block and
quantize the floating point value, and then dequantize it back and subtract
the original value to find the error. The error is then weighted by the
weight of the value. Note that this is using the simple linear mapping that we
would see when creating the block_q4_1, so there is nothing special happeing
yet, like nothing to do with the constraints of the scale and min values that we
mentioned earlier, but this is to have something to compare with:
```c++
    float iscale = nmax/(max - min); // quantized = iscale * (orig - min)
    float scale = 1/iscale;          // reconstructed = scale * quantized_value + min
    float best_mad = 0;              // should perhaps be best_error instead?
    for (int i = 0; i < n; ++i) {
        int l = nearest_int(iscale*(x[i] - min)); // calculate quantized value
        L[i] = MAX(0, MIN(nmax, l)); // clamp to 0..nmax
        float diff = scale * L[i] + min - x[i]; // reconstruct/dequantize the value and calculate the error
                                                   by  subtracting the original value
        diff = use_mad ? fabsf(diff) : diff * diff; // either use MAD or squared error
        float w = weights[i]; // get the weight for the current value
        best_mad += w * diff; // apply the weight to the error
    }
```
The variable name `best_mad` feels a bit misleading as if Mean Absolute Deviation
(MAD) is not used, which is the case for us, the it it is really just the
weighted quared error for a simple linear mapping.
```console
(gdb) p iscale
$21 = 1.07142854

(gdb) p scale
$22 = 0.933333337
```
Next there is a check if the number of steps is less than 1 in which case we
simple return the scale calculated which would be the same as the normal
block_q4_1:
```c++
    if (nstep < 1) {
        *the_min = -min;
        return scale;
    }
```
Next we will iterate over all the steps specified and see if we can improve
the base line quantization that we calculated above.
Now, some of the values that will be used are:
```console
(gdb) p nstep
$23 = 20
(gdb) p min
$24 = 0
(gdb) p rmin
$25 = -1
(gdb) p rdelta
$26 = 0.100000001
(gdb) p nmax
$27 = 15
(gdb) p rmin
$28 = -1
(gdb) p rdelta
$29 = 0.100000001
(gdb) p nmax
$30 = 15
(gdb) p max
$31 = 14
(gdb) p min
$32 = 0
```
```c++
    for (int is = 0; is <= nstep; ++is) {
        iscale = (rmin + rdelta*is + nmax)/(max - min); // calculate the scale for this iteration
        float sum_l = 0, sum_l2 = 0, sum_xl = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale*(x[i] - min)); // quantize float using the current scale
            l = MAX(0, MIN(nmax, l));  // clamp to 0..nmax
            Laux[i] = l; // store the quantized value in Laux for the current step.
            float w = weights[i]; // get the weight for the current value
            sum_l += w*l; // sum of the quantized value weighted by the weight
            sum_l2 += w*l*l; // sum of the squared quantized value weighted by the weight
            sum_xl += w*l*x[i]; // sum of the quantized values weighted by the weight and the original value
        }
```
The `sum_l`, `sum_l2`, and `sum_xl` are sums that we need for the weighted
least squares calculation.
* sum_l: weighted sum of the quantized values
* sum_l2: weighted sum of the squared quantized values (like Σ(wx^2))
* sum_xl: weighted sum of the quantized values multiplied by the original values (like Σ(wxy))

Notice that Laux is the quantized values for the current step. And as we go
through the steps (will be the next section) if the error is less that the
current best then we will update L to have these quantized values.

```console
is = 0: iscale  = (-1.0 + 0.0 + 15)/14 = 14/14   = 1.000
is = 5: iscale  = (-1.0 + 0.5 + 15)/14 = 14.5/14 = 1.036
is = 10: iscale = (-1.0 + 1.0 + 15)/14 = 15/14   = 1.071 ← Close to baseline!
is = 15: iscale = (-1.0 + 1.5 + 15)/14 = 15.5/14 = 1.107
is = 20: iscale = (-1.0 + 2.0 + 15)/14 = 16/14   = 1.143
```
Notice that we are trying scales around the baseline 1.071, from lower 1.000
to higher 1.143.

Following that we have:
```c++
        float D = sum_w * sum_l2 - sum_l * sum_l;  // D = denominator for least squares
        if (D > 0) {
            float this_scale = (sum_w * sum_xl - sum_x * sum_l)/D;
            float this_min   = (sum_l2 * sum_x - sum_l * sum_xl)/D;
            if (this_min > 0) { // we need the minimum to be zero
                this_min = 0; // force to 0
                this_scale = sum_xl / sum_l2; // recalculate the scale
            }
            float mad = 0; // trail_error
            for (int i = 0; i < n; ++i) {
               float diff = this_scale * Laux[i] + this_min - x[i]; // dequantize the quantized value in Laux
               diff = use_mad ? fabsf(diff) : diff * diff; // compute the error, absolut or squared
               float w = weights[i]; // get the weight
               mad += w * diff; // weight the error
            }
            if (mad < best_mad) {
                for (int i = 0; i < n; ++i) {
                    L[i] = Laux[i]; // Set the quantized value.
                }
                best_mad = mad; // set the best error found so far
                scale = this_scale; // set the best scale found so far
                min = this_min; // set the best minimum found so far
            }
        }
```
And when that loop completes we return:
```
    *the_min = -min;
    return scale;
}
```
And recall that the quantized values are stored in `L`. So this will return
us to:
```c++
        for (int j = 0; j < QK_K/32; ++j) {
            //scales[j] = make_qkx1_quants(32, 15, x + 32*j, L + 32*j, &mins[j], 9, 0.5f);
            float sum_x2 = 0;
            for (int l = 0; l < 32; ++l) sum_x2 += x[32*j + l] * x[32*j + l];
            float av_x = sqrtf(sum_x2/32);
            for (int l = 0; l < 32; ++l) weights[l] = av_x + fabsf(x[32*j + l]);
->          scales[j] = make_qkx2_quants(32, 15, x + 32*j, weights, L + 32*j, &mins[j], Laux, -1.f, 0.1f, 20, false);
            float scale = scales[j];
            if (scale > max_scale) {
                max_scale = scale;
            }
            float min = mins[j];
            if (min > max_min) {
                max_min = min;
            }
        }
```
So we can see that the scale is saved in the scales array and the max scale is
updated if the new scale is larger than the current max scale. The same goes for
the min value.

Next we will quantize the scale an min values. Notice that 63 is uses as the
maximum value for the scale and min values, so we are going to quantize the
scales and mins to fit in 6 bits, which is the number of bits required to
represent 63.
```c++
        float inv_scale = max_scale > 0 ? 63.f/max_scale : 0.f; // 8 different scales (on per 32-element block)
        float inv_min   = max_min   > 0 ? 63.f/max_min   : 0.f; // 8 different mins (on per 32-element block)
        for (int j = 0; j < QK_K/32; ++j) { // we are going to quantize these to save space (QK_K/32 = 8)
            uint8_t ls = nearest_int(inv_scale*scales[j]); // quantize the current scale. ls = quantized level
            uint8_t lm = nearest_int(inv_min*mins[j]); // quantize the current min.
            ls = MIN(63, ls); // clamp the scale to 0..63
            lm = MIN(63, lm); // clamp the min to 0..63
            // So j can be 0..7, and we have 8 scales and mins and each needs
            // 6 bits, so we have 16*6 = 96 bits, which is 12 bytes. And recall
            // that scales is uint8_t scales[12]
            // 
            //  6-bit value: [5][4][3][2][1][0]
            //                └──┘ └─────────┘
            //               high 2   low 4
            if (j < 4) { // just store the scale and min directly in the scales array after each other
                y[i].scales[j] = ls; // scales[0-3] = full 8 bit values (but only 6 bits used)
                y[i].scales[j+4] = lm; // scales[4-7] = full 8 bit values (but only 6 bits used)
            // After first 4 blocks:
            // scales[0] = ls₀ (scale for block 0)
            // scales[1] = ls₁ (scale for block 1)
            // scales[2] = ls₂ (scale for block 2)
            // scales[3] = ls₃ (scale for block 3)
            // scales[4] = lm₀ (min for block 0)
            // scales[5] = lm₁ (min for block 1)
            // scales[6] = lm₂ (min for block 2)
            // scales[7] = lm₃ (min for block 3)
            } else { // j=4,5,6,7
                // Place the low 4 bits of both the scale and min in th
                y[i].scales[j+4] = (ls & 0xF) | ((lm & 0xF) << 4); // uses all 8 bits
                y[i].scales[j-4] |= ((ls >> 4) << 6); // add to previous top bits
                y[i].scales[j-0] |= ((lm >> 4) << 6); // add to previous top bits
            }
        }
        y[i].d = GGML_FP32_TO_FP16(max_scale/63.f); // store the delta/scale so that the scales can be dequantized
        y[i].dmin = GGML_FP32_TO_FP16(max_min/63.f); // store the min delta/scale so that the mins can be dequantized.
```
Now, notice that we started with 8 32-bit floats scales = 32 bytes.
And we had 8 32-bit float mins = 32 bytes.
And we ended up with 12 bytes for all quantized scales and mins, plus 4 bytes
for the dequantization (d and dmin) giving us a total of 16 bytes!
The nice thing with the bit packing is that the 2 high bits of the later scale
and min values are placed in the unused bits of the first 4 entries (remember
that each entry is 8 bits but we only need 6 for one quantized value).

So we have the quantified the scales and mins and packed them into the scales
array, and we have the dequantize information in d and dmin. What is remaining
not is to store the quantified floating point values in the `q` member of the
block:
```c++
        uint8_t sc, m;
        for (int j = 0; j < QK_K/32; ++j) {
            get_scale_min_k4(j, y[i].scales, &sc, &m);
            const float d = GGML_FP16_TO_FP32(y[i].d) * sc; // dequantize scale delta
            if (!d) continue;
            const float dm = GGML_FP16_TO_FP32(y[i].dmin) * m; // dequantize min delta
            for (int ii = 0; ii < 32; ++ii) {
                int l = nearest_int((x[32*j + ii] + dm)/d); // quantize the float value (level)
                l = MAX(0, MIN(15, l)); // clamp to 0..15
                L[32*j + ii] = l; // store the quantized value in the L array (8 bits per entry)
            }
        }

        uint8_t * q = y[i].qs; // now we store the quantized values in the q member of the block
        for (int j = 0; j < QK_K; j += 64) {
            for (int l = 0; l < 32; ++l) {
                q[l] = L[j + l] | (L[j + l + 32] << 4); // pack the quantized values from L into q
                                                        // each entry in L is 8 bits but only hold
                                                        // 4 bits of the quantized value, so we can
                                                        // pack two quantized values into one byte.
            }
            q += 32;
        }

        x += QK_K;
```
And that is the complete quantization for the `block_q4_K` type.

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

`qs` is where the quantized values are stored.
So we have a array of 16 elements and notice the type is `uint8_t` which is 1
byte so each entry can hold 8 bits. Now each quantized value is 4 bits so we can
store two in each entry in this array:
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
With this we can store 32 quantized values in this array.
`qh` (h for high) is used to store the 5th bit of the quantized value so this is
16 bits in total, one for each entry in `qs` (the quantized values).

Like before we first need to calculate the delta:
```
delta = max_value / 31
```
This time we use 31 because we have 5 bits (11111b, 31d).
Then to get the quantized values we use the formula:
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

And to dequantize we use the formula:
```console
dequantized_value = quantized_value * delta

(where the quantized value is reconstructed from the nibbles and the 5th bit)

delta = 0.0161

12 (01100b): nibble=1100b, 5th_bit=0 → 01100b = 12 → 12 * 0.0161 = 0.193 ≈ 0.2
18 (10010b): nibble=0010b, 5th_bit=1 → 10010b = 18 → 18 * 0.0161 = 0.290 ≈ 0.3
25 (11001b): nibble=1001b, 5th_bit=1 → 11001b = 25 → 25 * 0.0161 = 0.403 ≈ 0.4
31 (11111b): nibble=1111b, 5th_bit=1 → 11111b = 31 → 31 * 0.0161 = 0.499 ≈ 0.5
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
```
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
far the values are all positive, like for 4 bits we have 0000b-1111b
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

###  TQ1_0 (Ternary Quantiztion
The `1` I think stands for the floor of the bits per weight which is floor(1.6875bpw = 1).
And `0` has the same meaning as for the other quantization types, like `q4_0`, `q5_0` and means
that there is only one delta value for the quantization and no minimum value.

Like the other types we can find the definition in `ggml-common.h`:
```c++
#define QK_K 256

// 1.6875 bpw
typedef struct {
    uint8_t qs[(QK_K - 4 * QK_K / 64) / 5]; // 5 elements per byte (3^5 = 243 < 256)
    uint8_t qh[QK_K/64]; // 4 elements per byte
    ggml_half d;
} block_tq1_0;
```
`QK_K` is 256 and the size of `qs` will be (256-4 * 256/64) / 5 = 48 bytes. And notice that
we also have `qh` which is 4 elements per byte, so we have 64/4 = 16 bytes for `qh` which gives
us a total of 48 + 16 + 2 = 66 bytes.

Here we have 3 quantization levels (0, 1, 2) and we can represent 3^5 = 243 values in 5 bits.
So this base 3:
```
-1  -> 0
 0  -> 1
 1  -> 2

3⁰ = 1
3¹ = 3
3² = 9
3³ = 27
3⁴ = 81
3⁵ = 243
```
So like the other quantization methods we've seen before this also takes an array of floating point
values, and a block to store the quantized values in:
```c++
void quantize_row_tq1_0_ref(const float * GGML_RESTRICT x, block_tq1_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int64_t i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max

        for (int j = 0; j < QK_K; j++) {
            const float v = x[j];
            amax = MAX(amax, fabsf(v));
        }

        const float d = amax;
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);
```
So the above will iterate over all 256 elements all blocks (nb is number of blocks).
And this first for looop will go through all the floating point values to get the
largest absolute max. This value will be used as the delta/scale and stored in d.
The we have the inverse of the delta `id` which allows us to use multiplication
instead of division in the computations. But as before the delta, not the inverse, is
what is stored in the block.
In our case `amax` will be 14.
```console
(lldb) p amax
(float) 14
(lldb) p id
(const float) 0.0714285746
```

Next we have the following loop which will:
```c++
        // 5 elements per byte, along 32 bytes
        for (size_t j = 0; j < sizeof(y->qs) - sizeof(y->qs) % 32; j += 32) {
            for (size_t m = 0; m < 32; ++m) {
                uint8_t q = 0;
                for (size_t n = 0; n < 5; ++n) {
                    int xi = lroundf(x[m + n*32] * id) + 1; // -1, 0, 1 -> 0, 1, 2
                    q *= 3;
                    q += xi;
                }
                // ceiling division (243 == pow(3, 5))
                q = ((uint16_t)q * 256 + (243 - 1)) / 243;
                y[i].qs[j + m] = q;
            }
            x += 5*32;
        }
```
```console
(lldb) p sizeof(y->qs)
(unsigned long) 48
(lldb) p sizeof(y->qs) % 32
(unsigned long) 16
(lldb) p sizeof(y->qs) - sizeof(y->qs) % 32
(unsigned long) 32
```
Lets break this down a bit, first we have the quantiztion using the inverse delta:
```c++
    (x[m + n*32] * id)
```
So this will produce a value between -1 and 1, and then we round it to the nearest
integer and then add 1 to convert it to unsigned (so -1 will become 0, 0 will become 1, and
1 will become 2):
```console
(lldb) p x[m + n*32] * id
(float) 0.714285731
(lldb) p (long) lround(x[m + n*32] * id)
(long) 1
(lldb) p (long) lround(x[m + n*32] * id) + 1
(long) 2
(lldb) p xi
(int) 2
```
One thing to notice is that this is doing `n*32` each time through the loop. So this is processing
with a stride of 32. Now, we need to keep in mind how the weight are used.
```console
input: [i0, i1, i2, i3]  (1×4 vector)

weights matrix (4×4):
Row 0: [w00, w01, w02, w03]
Row 1: [w10, w11, w12, w13]
Row 2: [w20, w21, w22, w23]
Row 3: [w30, w31, w32, w33]
```

Each vector multiplication (I’m thinking of the vector x matrix multiplication as separate individual
operations here) will need to load column of the weights. In my mind I see the operation like this:
the input vector is like a function that takes 4 parameters.
```
    [i0, i1, i2, i3]  [w00]
                      [w10]
                      [w20]
                      [w30]
```
So we call the function with the first column of the matrix to get the first output.
And by storing the weight with a spacing we can perform one load and get the weight's we
need for the function call..

For `TQ1_0` with stride 32:
```
x[m + 0*32] = x[m]     // Weight from "row" 0
x[m + 1*32] = x[m+32]  // Weight from "row" 32
x[m + 2*32] = x[m+64]  // Weight from "row" 64
x[m + 3*32] = x[m+96]  // Weight from "row" 96
x[m + 4*32] = x[m+128] // Weight from "row" 128
```
These 5 weights (originally separated by 32 positions) get packed into the same quantized byte
because they'll be needed together during one SIMD operation.

These 5 weights (originally separated by 32 positions) get packed into the same quantized byte
because they'll be needed together during one SIMD operation.

So we have now calculated the packed quantized value `q`:
```c++
                for (size_t n = 0; n < 5; ++n) {
                    int xi = lroundf(x[m + n*32] * id) + 1; // -1, 0, 1 -> 0, 1, 2
                    q *= 3;
                    q += xi;
                }
                // ceiling division (243 == pow(3, 5))
->              q = ((uint16_t)q * 256 + (243 - 1)) / 243;
                y[i].qs[j + m] = q;
```
So prior to the above marked line q is:
```console
(lldb) p (int) q
(int) 202
```
This is in base 3, so the range is [0, 242] (3^5 - 1). But a byte can store [0, 255] (8 bits), and this
means that 13 bits are unused (243-255).
```c++
new_q = (old_q * 256 + 242) / 243;
```
So first the current q value is scaled to the range [0, 255] by multiplying it with 256.
Then we add 242 to ensure that we round up instead of truncating. This is doing a ceiling
division by dividing by 243 (the +242/243 part).
```console
(lldb) p  ((q * 256) + 242) / 243
(int) 213

(lldb) p  ((242 * 256) + 242) / 243
(int) 255
```
One advantage of this is when we want to unpack a digit we can do the following:
```console
(lldb) expr int $packed = 255
(lldb) p $packed
(int) 255
(lldb) p ($packed * 243) / 256
(int) 242
(lldb) p (213 * 243) / 256
(int) 202
```
So the purpose of this is to be able to efficiently pack and unpack the quantized values. Using
division by 256 is a fast bitshift operation so unpacking will be very fast.
The last thing that happens is that x is incremented by `5*32`, so we move to the next set of
floating point values.

Next we have another loop which is very similar to the previous one except that this
time we are using a stride of 16 instead of 32. Now, in our case `y->qs` is 48 bytes, the
previous loop processed 32 element. And there are no more increments of 32 left so this is going
to process the remaining which is 48-32 = 16 elements. So notice that this is setting j = 32,
before the loop starts.
```c++
        // along 16 bytes
        for (size_t j = sizeof(y->qs) - sizeof(y->qs) % 32; j < sizeof(y->qs); j += 16) {
            for (size_t m = 0; m < 16; ++m) {
                uint8_t q = 0;
                for (size_t n = 0; n < 5; ++n) {
                    int xi = lroundf(x[m + n*16] * id) + 1; // -1, 0, 1 -> 0, 1, 2
                    q *= 3; // multiple by the base
                    q += xi;
                }
                // ceiling division (243 == pow(3, 5))
                q = ((uint16_t)q * 256 + (243 - 1)) / 243;
                y[i].qs[j + m] = q;
            }
            x += 5*16;
        }
```
And lastly we increment `x` by `5*16` which is 80 bytes.

Next we are going to populate the `qh` array.
```c++
        // 4 elements per byte
        for (size_t j = 0; j < sizeof(y->qh); ++j) {
            uint8_t q = 0;
            for (size_t m = 0; m < 4; ++m) {
                // -1, 0, 1 -> 0, 1, 2
                int xi = lroundf(x[j + m*sizeof(y->qh)] * id) + 1;
                q *= 3; // multiple by the base
                q += xi;
            }
            // shift the first value to the most significant trit (Ternary digit)
            q *= 3;
            // ceiling division (243 == pow(3, 5))
            q = ((uint16_t)q * 256 + (243 - 1)) / 243;
            y[i].qh[j] = q;
        }
        x += 4*sizeof(y->qh);
```
So this looks pretty much like the previous loop but this time we are using a stride of 4 and
notice that we have an additional multiplication by 3 before the division.

So lets just step back and see where we are:
First loop processed with stride of 32, j = 0, and processed 32 bytes
```
x += 5*32 = 160 elements
```
Second loop processed with stride of 16, j = 32, and processed 16 bytes
```
x += 5*16 = 80 elements
```
Total so far is 160 + 80 = 240 elements processed.
Remaining elements is 256 - 240 = 16 elements.

Third loop processed with stride of 4, and processes 4 bytes.
```
j = 0, x[0], x[4],  x[8], x[12] -> qh[0]
j = 1, x[1], x[5],  x[9], x[13] -> qh[1]
j = 2, x[2], x[6], x[10], x[14] -> qh[2]
j = 3, x[3], x[7], x[11], x[15] -> qh[3]
```
So this is packing the last 16 elements into the `qh` array which is 4 bytes. And this is using
a different packing that than `qs` array. So `qh` is the final 16 elements of the 256-element
weight vector.

Next we have:
```c++
    q *= 3;
```
So what is happening here is that are multiplying the packed base-3 number (ternary digit) by 3. Just think about
what this would do in base 10:
```
number = 42
(4 * 10¹) + (2 * 10⁰)

If we multiply this by 10 (the base) we get:
(4 x 10²) + (2 x 10¹) + (0 x 10⁰)

(4 x 100) + (2 x 10) + (0 x 1) = 420
```
Notice how the digits get shifted to the left and that we got a new digit at the end which is 0.

We are doing the same thing here but in base 3:
```
               2, 0, 1, 2
                   ↓
(2 * 3³) + (0 * 3²) + (1 * 3¹) + (2 * 3⁰) = 
                   ↓
(2 * 27) + (0 * 9) + (1 * 3) + (2 * 1) = 54 + 0 + 3 + 2 = 59
```
So this is a 4 trit number represented as an int 59. And now we shift it by multiplying it by
the base (3):
```
q_new = q * 3
3⁴ (81)	  3³ (27)	3² (9)	 3¹ (3)	   3⁰ (1)
  2         0         1        2        0 

This now becomes a 5 trit number:
q_new = (2 * 81) + (0 * 27) + (1 * 9) + (2 * 3) + (0 * 1)
q_new =   162    +    0     +    9    +    6    +    0    = 177
```
And then the same operation is performed as before:
```c++
    q = ((uint16_t)q * 256 + (243 - 1)) / 243;
```

Now looking at the dequantization:
```c++
         for (size_t n = 0; n < 4; ++n) {
             for (size_t j = 0; j < sizeof(x->qh); ++j) {
                 uint8_t q = x[i].qh[j] * pow3[n];
                 int16_t xi = ((uint16_t) q * 3) >> 8;
                 *y++ = (float) (xi - 1) * d;
             }
         }
```
We have 16 floats left to process, and the gh array which holds quantized data is 4 bytes long. So
n is which trit to extract, and j is the index of the 0-3 bytes of gh.
And for each element we are multiplying by a power of 3:
```c++
    const uint8_t pow3[6] = {1, 3, 9, 27, 81, 243};
```
`qh[j]` will just be an integer like 187 which is represents 5 packed trit values (with the last one just
zero and will be ignored).

The pow3 multiplication is used to move the trit of interest to be the most significant trit (t⁴). So the
first thing we to is isolate the trit of interest:
```c++
2162                 uint8_t q = x[i].qh[j] * pow3[n];
```
Recall from above that this is like multiplying by the base which shifts the trit of interest to the left,
doing this once, multiplying by 3 will move one step, and multiplying by 9 will move two steps, and so on.

And next we extract the most significant trit (t⁴) from the byte q:
```c++
                     int16_t xi = ((uint16_t) q * 3) >> 8;
```
We have q which is a byte which is the quantized values that we packed. The >> 8 is the same
as dividing by 256. So this is equivalent to floor(q * 3 / 256).

So what is happening is something like this:
```
most_sig_trit(v) = (v * 3) >> 8
remainder(v)     = (uint8_t)(v * 3)

Let's trace q = 187:


n = 0 (Extracting t₄):
q_shifted = 187 * pow3[0] = 187 * 1 = 187
xi = most_sig_trit(187)
xi = (187 * 3) >> 8 = 561 >> 8 = 2.

We extracted t₄ = 2. Correct.
The conceptual "remainder" for the next step is remainder(187) = (uint8_t)(187 * 3) = 49.

n = 1 (Extracting t₃):
Here's the trick:
q_shifted = 187 * pow3[1] = 187 * 3
Due to the uint8_t cast, this becomes:
(uint8_t)(561) = 49.
This is the exact same value as the "remainder" from the previous step.
xi = most_sig_trit(49)
xi = (49 * 3) >> 8 = 147 >> 8 = 0.
We extracted t₃ = 0. Correct.

The "remainder" is remainder(49) = (uint8_t)(49 * 3) = 147.

n = 2 (Extracting t₂):
q_shifted = (uint8_t)(187 * pow3[2]) = (uint8_t)(187 * 9) = (uint8_t)(1683) = 147.
Again, this is the remainder from the prior step.
xi = most_sig_trit(147)
xi = (147 * 3) >> 8 = 441 >> 8 = 1. We extracted t₂ = 1. Correct.

n = 3 (Extracting t₁):
q_shifted = (uint8_t)(187 * pow3[3]) = (uint8_t)(187 * 27) = (uint8_t)(5049) = 185.
xi = most_sig_trit(185)
xi = (185 * 3) >> 8 = 555 >> 8 = 2. We extracted t₁ = 2. Correct.
```

And the first iteration will multiply 187 by 1:
```
(lldb) p (187 * 1) * 3 >> 8
(int) 2
      ↓
      2 0 1 2 0
```
Now for the next iteration we will multiply by 3, but notice that we are storing the result in an
unsigned 8 bit integer so we only have 8 bits, that is 0-255, so this will overflow (which is alright
and not undefined behavior like it would be for signed integers):
```console
(lldb) p 187 * 3
(int) 561
(lldb) p/u 561 % 256
(int) 49

(lldb) p/u (uint8_t) (187 * 3)
(uint8_t) 49
(lldb) p/u (uint8_t) (49 * 3) >> 8
(int) 0
      ↓
    2 0 1 2 0
```
```console
(lldb) p/u (uint8_t)(187 * 9)
(uint8_t) 147

(lldb) p (uint16_t)(147 * 3) >> 8
(int) 1
      ↓
  2 0 1 2 0
```
```console
(lldb) p/u (uint8_t)(187 * 27)
(uint8_t) 185
(lldb) p (uint16_t)(185 * 3) >> 8
(int) 2
      ↓
2 0 1 2 0
```
Or a little more compact:
```console
(lldb) p/u (uint8_t) (187 * 1) * 3 >> 8
(int) 2
(lldb) p/u (uint8_t) (187 * 3) * 3 >> 8
(int) 0
(lldb) p/u (uint8_t) (187 * 9) * 3 >> 8
(int) 1
(lldb) p/u (uint8_t) (187 * 27) * 3 >> 8
(int) 2
```
And we only process 4 elements so this will simply skip the last 0 trit.
