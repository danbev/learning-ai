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
delta/scale and the minimum value the K block quantizes these values. So there
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
y axis like we did perviously, and on the x axis we have the quantized units.

What we want to do is to find a line that fits these points well and we can
use linear regression/least squares to find the best fit line. The slope of this
line is the scale and the y-intercept is the minimum value. But we also have
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

So d/d(scale) is the derivative operator/function that takes (y_i - scale * x_i + min))²
as its input.

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
```c++
void quantize_row_q4_K_ref(const float * GGML_RESTRICT x, block_q4_K * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    uint8_t L[QK_K];
    uint8_t Laux[32];
    float   weights[32];
    float mins[QK_K/32];
    float scales[QK_K/32];

    for (int i = 0; i < nb; i++) {
        float max_scale = 0; // as we are deducting the min, scales are always positive
        float max_min = 0;
        // QK_K is 256, so we have 8 blocks of 32 elements each. j's range 0-8.
        for (int j = 0; j < QK_K/32; ++j) {

            float sum_x2 = 0;
            for (int l = 0; l < 32; ++l) {
                sum_x2 += x[32*j + l] * x[32*j + l];
            }

            float av_x = sqrtf(sum_x2/32);
            for (int l = 0; l < 32; ++l) {
                weights[l] = av_x + fabsf(x[32*j + l]);
            }
            scales[j] = make_qkx2_quants(32, 15, x + 32*j, weights, L + 32*j, &mins[j], Laux, -1.f, 0.1f, 20, false);
            float scale = scales[j];
            if (scale > max_scale) {
                max_scale = scale;
            }
            float min = mins[j];
            if (min > max_min) {
                max_min = min;
            }
        }

        float inv_scale = max_scale > 0 ? 63.f/max_scale : 0.f;
        float inv_min   = max_min   > 0 ? 63.f/max_min   : 0.f;
        for (int j = 0; j < QK_K/32; ++j) {
            uint8_t ls = nearest_int(inv_scale*scales[j]);
            uint8_t lm = nearest_int(inv_min*mins[j]);
            ls = MIN(63, ls);
            lm = MIN(63, lm);
            if (j < 4) {
                y[i].scales[j] = ls;
                y[i].scales[j+4] = lm;
            } else {
                y[i].scales[j+4] = (ls & 0xF) | ((lm & 0xF) << 4);
                y[i].scales[j-4] |= ((ls >> 4) << 6);
                y[i].scales[j-0] |= ((lm >> 4) << 6);
            }
        }
        y[i].d = GGML_FP32_TO_FP16(max_scale/63.f);
        y[i].dmin = GGML_FP32_TO_FP16(max_min/63.f);

        uint8_t sc, m;
        for (int j = 0; j < QK_K/32; ++j) {
            get_scale_min_k4(j, y[i].scales, &sc, &m);
            const float d = GGML_FP16_TO_FP32(y[i].d) * sc;
            if (!d) continue;
            const float dm = GGML_FP16_TO_FP32(y[i].dmin) * m;
            for (int ii = 0; ii < 32; ++ii) {
                int l = nearest_int((x[32*j + ii] + dm)/d);
                l = MAX(0, MIN(15, l));
                L[32*j + ii] = l;
            }
        }

        uint8_t * q = y[i].qs;
        for (int j = 0; j < QK_K; j += 64) {
            for (int l = 0; l < 32; ++l) {
                q[l] = L[j + l] | (L[j + l + 32] << 4);
            }
            q += 32;
        }

        x += QK_K;
    }
}
```

_wip_


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

### Bits Per Weight (BPW)
