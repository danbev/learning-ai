## GGML Quantization
In broad terms this is about taking a floating point number like a single
precision floating point number (32 bits) or a half precision floating point
number (16 bits) and converting it to a fixed point number with a fixed number
of bits. 

* QI is the quantized value
* QK is the number of bits used for quantization
* QR is the ratio of the quantized value and the number for which it is a quantization(?)

### `block_q4_0`
This struct is defined in ggml/src/ggml-quants.h:
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
delta = max_value / number of bits
delta = max_value / 15                (1111b)
```

`qs` is where are quantized values are stored. So we have a array of 16 elements
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
packed into `ggml_half2` I think) member. The `m` member is used to store the
smallest value in the block of float values.

So this is a common patterns where `_0` there is only the one delta value but
with `_1` there is an additional field to store data which in this case is the
minimum.

#### Quantization
Calculate min and delta using the forumla:
```
delta = (max_value - min_value) / 15
```
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
Notice that `qa` is in fact the same size as for `block_q4_0` but we have an
additional field `qh` which has an array of 5 (0-4). This is used to store the
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
