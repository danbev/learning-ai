## The Gemma QAT Method
This is code that could possibly be used for Gemma 3 270 QAT models.

Instead of using the extermal value, it calculates the root mean square. The
standard impl is based on outliers, the max value.

This is a learned quantization scheme where the scalar multiplier was optimized
during training. So the `scale_multiplier` is not a constant value, but rather
a learned value that was determined during the quantization-aware training (QAT)
process and could be different for different models or configurations. Where
can I find this? From the model configuration?


### The problem with max-based scaling
```console
Weights: [-0.8, -0.3, 0.1, 0.4, 0.7, -0.2, 0.9, -1.2, 0.6, -0.5, 0.3, -0.1,
          0.8, -0.4, 0.2, -0.6, 1.1, -0.9, 0.5, -0.7, 0.0, 0.35, -0.15, 0.25,
          -0.45, 0.65, -0.25, 0.15, -0.35, 0.55, -0.05, 4.8]  // 32 values, one outlier

max = 4.8  (the outlier dominates)
scale = 4.8 / -8 = -0.6
id = 1.0 / -0.6 = -1.667

Most values end up as:
-0.8 × -1.667 = 1.33   →   1.33 + 8.5 = 9.83 → stored as 10
-0.3 × -1.667 = 0.5    →    0.5 + 8.5 = 9.0  → stored as 9  
0.1  × -1.667 = -0.167 → -0.167 + 8.5 = 8.33 → stored as 8
0.4  × -1.667 = -0.667 → -0.667 + 8.5 = 7.83 → stored as 8
... (most cluster around 7-10)
4.8  × -1.667 = -8.0   →   -8.0 + 8.5 = 0.5  → stored as 0
```
Values are clustered in a narrow range (7-10), wasting the full 0-15 range.

With the RMS-based scaling, the extreme value is not used to determine the scale,
but considers the distribution of all values, leading to a more balanced quantization.
```console
RMS = sqrt(sum of squares / 32) ≈ 1.1  (considers all values, not dominated by outlier)
scale = 0.37755 × 1.1 = 0.415
id = 1.0 / 0.415 = 2.41

Now we get better spread:
-0.8 × 2.41 = -1.93 → -1.93 + 8.0 = 6.07 → stored as 6
-0.3 × 2.41 = -0.72 → -0.72 + 8.0 = 7.28 → stored as 7
0.1 × 2.41 = 0.24 → 0.24 + 8.0 = 8.24 → stored as 8
0.4 × 2.41 = 0.96 → 0.96 + 8.0 = 8.96 → stored as 9
0.7 × 2.41 = 1.69 → 1.69 + 8.0 = 9.69 → stored as 10
... (values spread across 3-13 range)
4.8 × 2.41 = 11.57 → 11.57 + 8.0 = 19.57 → clamped to 15
Weights: [0.1, 0.2, 0.15, 0.18, 0.12, 0.16, 0.14, 8.7]
```

RMS method uses more of the available quantization range because:

* Max method: Scale is determined by the single largest value, forcing
  everything else into a narrow band
* RMS method: Scale reflects the "typical" magnitude of weights, spreading them
  across more quantization levels

### Implementation

```c++
// reference implementation for deterministic creation of model files
void quantize_row_q4_0_ref(const float * GGML_RESTRICT x, block_q4_0 * GGML_RESTRICT y, int64_t k) {
    static const int qk = QK4_0;

    assert(k % qk == 0);

    const int nb = k / qk;

#ifdef GGML_USE_GEMMA_QAT_QUANTS
    const float scale_epsilon = 1e-12f; // tiny constant to avoid division by zero
    const float scale_multiplier = 0.37755f; // Model specific scaling factor
    const float maxq = 15.0f; // maximum quantized value (4 bits = 0-15)
    const float zero = maxq/2.0f + 0.5f; // zero point 15/2.0 + 0.5 = 8.0.

    // First we create sum of squares to calculate the RMS scale
    float mean = 0.0f;
    for (int i = 0; i < k; i++) {
        mean += x[i] * x[i]; // sum of squares
    }
    // This is the RMS calculation
    mean /= k; // mean of squares

    // This is what will be used to scale the values to be quantified. So
    // instead of using delta (or inverse delta) we use this scale value
    // which is the RMS value scaled by a learned multiplier.
    float scale = scale_multiplier * sqrtf(mean); // RMS * 0.37755f
    scale += scale_epsilon; // prevent division by zero

    // Now iterate over all the blocks and quantize them
    for (int i = 0; i < nb; i++) {
        // store the scale for dequantization
        y[i].d = GGML_FP32_TO_FP16(scale);

        // Process values in pairs, qk/2 = 32/2 = 16 iterations
        // So each iteration will process two float values and pack them into 1 byte.
        for (int j = 0; j < qk/2; ++j) {
            // scale/divide value in x for positions 0, 1, 2, 3, ... ,15
            const float x0 = x[i*qk + 0    + j] / scale;
            // scale/divide value in x for positions 16, 17, 18, 19, .. ,31
            const float x1 = x[i*qk + qk/2 + j] / scale;

            // Quantize x0: add zero point (8.0) to center the range, round to nearest
            // integer, then clamp to valid 4-bit range [0, 15]
            const uint8_t xi0 = MAX(0, MIN(maxq, roundf(x0 + zero))); // adding 8.0
            // Quantize x1: add zero point (8.0) to center the range, round to nearest
            // integer, then clamp to valid 4-bit range [0, 15]
            const uint8_t xi1 = MAX(0, MIN(maxq, roundf(x1 + zero))); // adding 8.0

            // Store xi0 (not uint8_t) in lower 4 bits
            y[i].qs[j]  = xi0;
            // Store xi1 (not uint8_t) in upper 4 bits
            y[i].qs[j] |= xi1 << 4;
        }
    }
#else
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
#endif
}
```
The standard scales based on the max absolute value but the gemma qat scale is
based on the Root Mean Squared (RMS) * 0.33755f.

The 0.37755 multiplier was likely learned during QAT training

So notice that above the packing of quantized values is something like this:
```console
x 
  [f0, f1, f2, f3, f4, f5, f6, f7, f8, f0, f10, f11, f12, f13, f14, f15, f16, f17... f31]

    low  high
qa  [f0, f16]
    [f1, f17]
    [f2, f18]
    [f3, f19]
     ...
    [f15, f31]
```
Notice that in the low bits we have the first quantized values, and in the high
bits we have the second quantized values.

So a single 16-byte load holds all 32 4-bit codes: the first 16 in the low
nibbles and the second 16 in the high nibbles.

The advantage of this packing is that it allows efficient processing, as we can
then do:
```c++
uint8_t b0 = qs[0];
uint8_t low0 = b0 & 0x0F;
```
And that would give the low nibble of the first byte.

Now, with SIMD we can do the following:
```console
__m128i v   = _mm_loadu_si128((const __m128i*)qs);      // loads qs[0..15]
__m128i msk = _mm_set1_epi8(0x0F);                      // mask = 0x0F repeated in all bytes
__m128i lo  = _mm_and_si128(v, msk);
```
This does the same things as our c code above but for all 16 bytes in v get
masked in parallel — no loop needed, no single-value restriction.
This would be like the following in c++:
```c++
for (int i = 0; i < 16; i++) {
    lo[i] = qs[i] & 0x0F;
}
```

