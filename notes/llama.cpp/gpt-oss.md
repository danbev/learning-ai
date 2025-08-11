## OpenAI GPT OSS
These are open source models that OpenAI has released.

### Converting to GGUF
First clone the model from HuggingFace:
```console
$ git clone https://huggingface.co/openai/gpt-oss-20b
```

Then use the `convert_hf_to_gguf.py` script to convert the model:
```console
(venv) $ python convert_hf_to_gguf.py /home/danbev/work/ai/models/gpt-oss-20b --outfile models/gpt-oss-20b.gguf
...
INFO:gguf.gguf_writer:Writing the following files:
INFO:gguf.gguf_writer:gpt-oss-20b.gguf: n_tensors = 459, total_size = 13.8G
Writing: 100%|█████████████████████████████████████████████████████████████████████████████████| 13.8G/13.8G [01:27<00:00, 157Mbyte/s]
INFO:hf-to-gguf:Model successfully exported to models/gpt-oss-20b.gguf
```
Now, the model is not trained with quantization (not Quantize-Aware Trained (QAT))
but we can post-train quantize it in llama.cpp. So the question if more what type
to quantize to. Lets inspect the original model to see what types it is used
for the tensor that are most often quantized (like weights).
```console
```

### Model Details


### MXFP4
This is a 4-bit floating point format and the name comes from the Microscaling
FP 4 which is a standard created by the Open Compute Project (OCP).

Each value is stored in just 4 bits and this is often called `E2M1`
because it has 1 Sign bit, 2 `E`xponent bits, 1 `M`antissa bit):
```
[ S | E1 E0 | M ]
S = Sign bit (0 = positive, 1 = negative)
E = 2-bit exponent (biased)
M = 1-bit mantissa (fraction)
```
So have the sign bit which is just that the sign so we can ignore that for now.
We have two bits for the exponent, and one bit for the mantissa.
Now, floating point formats store the exponent offset from zero using a bias.
For `E2` a common bias is 1.
```
stored_exponent  actual_exponent
00               -1
01                0
10                1
11                special (zero/subnormal/NaN)
```
So only three of the exponent codes are “normal” values; the fourth (11) is
reserved for special cases
```console
value = (1 + M * 0.5) × 2^(actual_exponent)

where m is either 0 or 1
```

So if we have the exponent of 00 bit pattern which represents -1 and the
mantissa i 0 we get:
```
(1 + M * 0.5) × 2^(actual_exponent) = value
(1 + 0 * 0.5) * 2^(-1)              = 
            1 * 2^(-1)              = 
            1 * 0.5                 = 0.5
```
And the mantissa can also be 1:
```
(1 + 1 * 0.5) * 2^(-1)              = 
          1.5 * 0.5                 = 0.75
```

```
| E bits | M bit | Actual exponent (bias=1) | Significand | Result                 |
| ------ | ----- | ------------------------ | ----------- | ---------------------- |
| 00     | 0     | -1                       | 1.0         | 0.5                    |
| 00     | 1     | -1                       | 1.5         | 0.75                   |
| 01     | 0     | 0                        | 1.0         | 1.0                    |
| 01     | 1     | 0                        | 1.5         | 1.5                    |
| 10     | 0     | 1                        | 1.0         | 2.0                    |
| 10     | 1     | 1                        | 1.5         | 3.0                    |
| 11     | 0/1   | —                        | —           | Special: zero/NaN/etc. |
```
Now, notice that the largest exponent we can have the bit pattern 10, which
gives the value 1. And the largest mantissa is 1 which is 1.5.
```
(1 + 1 * 0.5) * 2^(1)              = 
          1.5 * 2                  = 3
```
So the largest value that we can represent is 3.

In IEEE-like floating point standards (and MXFP4 follows that structure), the
all-ones exponent code is reserved:
* All mantissa bits zero → infinity
* Non-zero mantissa → NaN

In quantized formats like MXFP4, they often don’t bother with infinities/NaNs, but still keep the mapping for compatibility and to simplify hardware decoding. Instead, they may treat E=11 as zero or ignore it entirely — but they still can’t use it as a “normal” number.

This is not much and with only this it might not be that useful. Now, if we recall
that quantization is a process where we map a range of values from a continuous
domain to a discrete set of values. To do this we have to scale the floating
point values in some way. There is more information about quantization in the
[ggml-quantization.md](../ggml/quantization.md) document. But the gist of it is that we
need a scale factor to convert which is not included in the above 4 bits, so
how does this work. Well, it looks like they are doing something similar to
what llama.cpp does for block quantization.

So what is done is that tensors are grouped into blocks of ofter 32 values, and
all values in the group share the same scale factor:
```console
  [   E8M1   ]
  [E2M1][E2M1][E2M1][E2M1]...[E2M1][E2M1][E2M1][E2M1]
    0                                            31
```
So lets say we have a large tensor, like a weight matrix:
```
[ 0.001, -0.45, 1.25, -21.4, ... ]  → Block 0 (32 values)
[ ... ]                           → Block 1 (32 values)

```
Then we need to find this blocks scaling factor. This is a E8M1 value which means
that it looks like this:

```
 +---+---+---+---+---+---+---+---+
 | E | E | E | E | E | E | E | M |
 +---+---+---+---+---+---+---+---+
```
Now, the scale is calculated as follows
```
scale = max(|x|) / max_representable_value_in_E2M1
```
Where `max_representable_value_in_E2M1` is the maximum value that can be
represented in E2M1 format. And as we saw above, the maximum value is 3.0.
So we take the maximum absolute value in the block and divide it by 3.0 to
get the scaling factor for the block. So if 50.0 was the maximum absolute value
we would get:
```console
scale = 50 / 3.0  ≈ 16.666...
```
The scale each value using:
```
scaled_value_i = original_value_i / scale

Example
scaled_value_1 = -0.45 / 16.666... ≈ -0.027
```
We then need to convert this value in to the E2M1 format:
```
     -0.027
      ↓
sign  1

abs(scaled_value) = 0.027.
```
The goal is to find the nearest representable E2M1 value from the table above
for this value. The positive normal values are:
```
0.5  0.75  1.0  1.5  2.0  3.0
```
0.027 is smaller than the smallest positive normal value (0.5), so it will be
rounded to 0.0 in E2M1:
```
E bits = 11  (zero/special code in MXFP4 usage)
M bit  = 0   (zero mantissa)
```
```
S | E1 E0 | M
1 |  1  1 | 0

Binary: 1110
```
Lets take a larger value as well, that does not quantized to the value zero as
the dequatization would not be very interesting:
```
original_value = -21.4
scaled_value = -21.4 / 16.666...
              ≈ -1.284
sign_bit = 1
abs_value = 1.284

0.5, 0.75, 1.0, 1.5, 2.0, 3.0

significand = 1.5 → M bit = 1

S | E1 E0 | M
1 |  0  1 | 1

Binary: 1011
```

To dequantize we can do:
```
X_i = P_i * scale

where P_i is the quantized value and scale is the scale factor for the block.
```
So for the above example we would take the quantized value of 0 and multiply
it by the scale factor:
```console
original_approx = = -1.5 × 16.666...
                 ≈ -25.0

true value  = -21.4
quantized   ≈ -25.0
error       ≈ -3.6
```

### MXFP4 in GGML
In ggml/ggml-common.h we can find:
```c++
#define QR_MXFP4 2
#define QI_MXFP4 (QK_MXFP4 / (4 * QR_MXFP4))

#define QK_MXFP4 32
typedef struct {
    uint8_t e; // E8M0
    uint8_t qs[QK_MXFP4/2];
} block_mxfp4;
```
SO we can see that the scale is stored in the `e` field which is an E8M0. Note
that this is does not have a mantissa so it is only an exponent and it stored
as an encoded exponent byte so it has to be run through an E8M0 decoder to get
the actual scale value. This is problably the reason for naming this field `e`.

```c++
#define QI_MXFP4 (QK_MXFP4 / (4 * QR_MXFP4))
#define QI_MXFP4 (      32 / (4 * 2))
#define QI_MXFP4 (      32 / 8)
#define QI_MXFP4 (      32 / 8)
#define QI_MXFP4 (           4)
```
And `qs` will stored the quantized E2M1 values in 4 `uint8_t` values.

```c++
// e2m1 values (doubled)
// ref: https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
GGML_TABLE_BEGIN(int8_t, kvalues_mxfp4, 16)
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
GGML_TABLE_END()
```
This is the same table we saw above but with the values doubled:
```
{ 0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6  }
                 ↓
{ 0,  ±1 , ±2,  ±3 , ±4, ±6, ±8, ±12 }
```
If floats were used for the lookup table, this would mean slower indexing and
and cost more space. So 

