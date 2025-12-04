### Temperature zero implementation explanation

```c++
    if (ctx_data->temp <= 0.0f) {
        // Find the most probable token index.
        struct ggml_tensor * max_idx = ggml_argmax(ctx, data->logits);
        ggml_set_name(max_idx, "temp_max_idx");

        // Reshape logits to 2D so we can use get_rows.
        struct ggml_tensor * logits_rows = ggml_reshape_2d(ctx, data->logits, 1, data->logits->ne[0]);
        ggml_set_name(logits_rows, "temp_logits_rows");

        // Get the max logit value.
        struct ggml_tensor * max_logit = ggml_get_rows(ctx, logits_rows, max_idx);
        ggml_set_name(max_logit, "temp_max_logit");

        // Repeat max_logit to match logits shape for element-wise operations.
        // We can't just to ggml_sub(ctx, max_logit, data->logits) cause of the broadcasting
        // rules in ggml. We can't broadcast [32000] to [1], but the other way round works.
        // [1] to [32000] is ok.
        struct ggml_tensor * max_logit_repeated = ggml_repeat(ctx, max_logit, data->logits);
        ggml_set_name(max_logit_repeated, "temp_max_logit_repeated");

        // Compute diff = max - logits.
        // At max_idx position this value will be zero, and positive elsewhere.
        struct ggml_tensor * diff = ggml_sub(ctx, max_logit_repeated, data->logits);
        ggml_set_name(diff, "temp_diff");

        // Subtract small epsilon to make argmax position negative.
        // This ensures ggml_step returns 0 at argmax across all backends.
        struct ggml_tensor * diff_eps = ggml_scale_bias(ctx, diff, 1.0f, -1e-6f);
        ggml_set_name(diff_eps, "temp_diff_eps");

        // Create mask: step gives 1 for non-max positions, 0 for max position
        struct ggml_tensor * mask = ggml_step(ctx, diff_eps);
        ggml_set_name(mask, "temp_mask");

        // Convert mask to bias: -1e9 for non-max, 0 for max
        const float large_val = 1e9f;
        struct ggml_tensor * bias = ggml_scale_bias(ctx, mask, -large_val, 0.0f);
        ggml_set_name(bias, "temp_bias");

        data->logits = ggml_add(ctx, data->logits, bias);
        ggml_set_name(data->logits, "temp_zero_logits");

        ggml_build_forward_expand(gf, data->logits);
        return;
    }
```
```console
(lldb) p max_idx->ne
(int64_t[4])  ([0] = 1, [1] = 1, [2] = 1, [3] = 1)

(lldb) p max_logit->ne
(int64_t[4])  ([0] = 1, [1] = 1, [2] = 1, [3] = 1)

(lldb) p max_logit_repeated->ne
(int64_t[4])  ([0] = 32000, [1] = 1, [2] = 1, [3] = 1)
```

```
logits (shape [10]):
[2.1, 5.3, 1.8, 7.2, 3.4, 4.1, 6.8, 2.9, 5.7, 4.5]
                 ↑                
              max logit

Max logit is 7.2 at index 3.
```
We want to zero out all other logits except the max logit.

```
struct ggml_tensor * max_idx = ggml_argmax(ctx, data->logits);
max_idx = 3  (scalar tensor containing the index)
[3]
```

```
struct ggml_tensor * logits_rows = ggml_reshape_2d(ctx, data->logits, 1, data->logits->ne[0]);

Original shape: [10]
New shape     : [1, 10]  (1 column, 10 rows)

logits_rows (no data copied, just view change):
 [2.1]
 [5.3]
 [1.8]
 [7.2]  <- row 3
 [3.4]
 [4.1]
 [6.8]
 [2.9]
 [5.7]
 [4.5]
```
```
struct ggml_tensor * max_logit = ggml_get_rows(ctx, logits_rows, max_idx);
Gets row 3 from logits_rows:
 [2.1]
 [5.3]
 [1.8]
 [7.2]  <- row 3
 [3.4]
 [4.1]
 [6.8]
 [2.9]
 [5.7]
 [4.5]
max_logit = [7.2]  (shape [1])
```

```
struct ggml_tensor * max_logit_repeated = ggml_repeat(ctx, max_logit, data->logits);

max_logit_repeated (shape [10]):
[7.2, 7.2, 7.2, 7.2, 7.2, 7.2, 7.2, 7.2, 7.2, 7.2]
```

```
struct ggml_tensor * diff = ggml_sub(ctx, max_logit_repeated, data->logits);

[7.2, 7.2, 7.2, 7.2, 7.2, 7.2, 7.2, 7.2, 7.2, 7.2]
[2.1, 5.3, 1.8, 7.2, 3.4, 4.1, 6.8, 2.9, 5.7, 4.5]

diff = max_logit_repeated - logits:
[7.2-2.1, 7.2-5.3, 7.2-1.8, 7.2-7.2, 7.2-3.4, 7.2-4.1, 7.2-6.8, 7.2-2.9, 7.2-5.7, 7.2-4.5]

diff:
[5.1, 1.9, 5.4, 0.0, 3.8, 3.1, 0.4, 4.3, 1.5, 2.7]
                 ↑
At max position: exactly 0.0
Everywhere else: positive values
```

```
struct ggml_tensor * diff_eps = ggml_scale_bias(ctx, diff, 1.0f, -1e-6f);

scale_bias(x, scale, bias) = scale * x + bias

So this operation does:
diff_eps = 1.0 * diff + (-0.000001)

diff_eps:
[5.099999, 1.899999, 5.399999, -0.000001, 3.799999, 3.099999, 0.399999, 4.299999, 1.499999, 2.699999]
                                 ↑
Now the max position is slightly NEGATIVE!
All others are still positive.
```

```
struct ggml_tensor * mask = ggml_step(ctx, diff_eps);
Apply step function (x < 0 → 0, x >= 0 → 1):

diff_esp:
[5.099999, 1.899999, 5.399999, -0.000001, 3.799999, 3.099999, 0.399999, 4.299999, 1.499999, 2.699999]

mask:
[   1,        1 ,       1,          0,       1,        1,        1,        1,       1,        1]
                                    ↑ 
Max position gets 0, everything else gets 1.
```

```
// where large_val = 1e9
struct ggml_tensor * bias = ggml_scale_bias(ctx, mask, -large_val, 0.0f);

bias = -1e9 * mask + 0.0

bias:
[-1e9, -1e9, -1e9, 0, -1e9, -1e9, -1e9, -1e9, -1e9, -1e9]
                   ↑

Max position gets 0, everything else gets -1 billion
```

```
data->logits = ggml_add(ctx, data->logits, bias);

logits:
[2.1,   5.3,  1.8,  7.2,  3.4,  4.1,  6.8,  2.9,  5.7,  4.5]
bias:
[-1e9, -1e9, -1e9,  0.0, -1e9, -1e9, -1e9, -1e9, -1e9, -1e9]


1e9  = 1,000,000,000 (one billion)
-1e9 = -1,000,000,000 (negative one billion)

So we are adding a very large negative number to every logit
except the max logit position where we are adding 0.0.

Element-wise addition:
Position 0: 2.1 + (-1e9) = 2.1 - 1000000000 = -999999997.9
Position 1: 5.3 + (-1e9) = 5.3 - 1000000000 = -999999994.7
Position 2: 1.8 + (-1e9) = 1.8 - 1000000000 = -999999998.2
Position 3: 7.2 + 0.0 = 7.2
Position 4: 3.4 + (-1e9) = 3.4 - 1000000000 = -999999996.6
... and so on

Final result:
[-999999997.9, -999999994.7, -999999998.2, 7.2, -999999996.6, ...]
```
Notice that all values become very large negative numbers except for the max logit!


### temp ext sampler
This is a dynamic temperature sampler and differnt from the above sampler which
just scales using a provided temp value.

This type of sampling looks at how uncertain the model currently is, using the
entropy of the logits.

We have the following parameters for this sampler:
```console
temp = 1.0
delta = 0.5
exponent = 1.0
```
And lets say we have our input logits:
```console
[2.1, 5.3, 1.8, 7.2, 3.4, 4.1, 6.8, 2.9, 5.7, 4.5]
```
Goal: Dynamically adjust temperature based on the entropy of the probability
distribution.

High entropy: "I don't know what will happen", the logits are for uniform,
a more flat distribution (if we think of them as columns in a bar chart).

Low entropy: "I'm pretty sure what will happen", the logits are peaked, so
this will be a more spiky distribution.

The maximum entropy for a distribution is where all outcomes are equally likely.
For example for our ten tokens it would be:
```
Uniform distribution:
[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
```
Every token has equal probability (10% each). This is maximum uncertainty we have
no idea which token will be selected.

```
entropy = -Σ(p_i * log(p_i))
        = -Σ(1/N * log(1/N))     [all p_i are 1/N]
        = -N * (1/N * log(1/N))  [N identical terms]
        = -log(1/N)
        = log(N)

max_entropy = log(10) ≈ 2.303
```


We calculate the min and max temperaturs as:
```c++
const float min_temp = std::max(0.0f, ctx->temp - ctx->delta);
```

So to translate these to ggml_operations:
```console
    // Perform dynamic temperature scaling.
    const float min_temp = std::max(0.0f, ctx_data->temp - ctx_data->delta);
    const float max_temp = ctx_data->temp + ctx_data->delta;
    const float max_entropy = logf(data->logits->ne[0]);

    // Calculate the probabilities. Can we get away without this?
    struct ggml_tensor * softmax = ggml_soft_max(ctx, data->logits);

    // Calculate the entropy.
```
Now we have to think about this, we have our input logits and we have the
probabilities after softmax:
```
logits:
[ 2.1,    5.3,    1.8,    7.2,    3.4,    4.1,    6.8,    2.9,     5.7,   4.5]
softmax:
[0.0028, 0.0679, 0.0021, 0.4542, 0.0102, 0.0205, 0.3044, 0.0062, 0.1013, 0.0305]
```
And we want to perform:
```c++
    float entropy = 0.0f;
    for (float p : probs) {
        if (p > 0.0f) {
            // So entropy is the probability times log(probability) and then
            // summed over all probabilities.
            entropy -= p * logf(p);
        }
    }
```
```
struct ggml_tensor * log_probs = ggml_log(ctx, softmax);
softmax:   [0.0028, 0.0679, 0.0021, 0.4542, 0.0102, 0.0205, 0.3044, 0.0062, 0.1013, 0.0305]
log_probs: [-5.88,  -2.69,  -6.17,  -0.79,  -4.59,  -3.89,  -1.19,  -5.08,  -2.29,  -3.49]
```
So each entry in softmax needs to be multiplied by the logf(p) of iself and then
summed up.

struct ggml_tensor * p_log_p = ggml_mul(ctx, softmax, log_probs);
p_log_p: [-0.0165, -0.1827, -0.0130, -0.3588, -0.0468, -0.0797, -0.3622, -0.0315, -0.2320, -0.1064]
```
The we sum:
```
struct ggml_tensor * sum_p_log_p = ggml_sum(ctx, p_log_p);
sum_p_log_p = -1.4296  (scalar tensor)


### Temperature zero implementation
```console
    if (temp <= 0.0f) {
        // Find the most probable token index.
        struct ggml_tensor * max_idx = ggml_argmax(ctx, data->logits);
        ggml_set_name(max_idx, "temp_max_idx");
        // This will have shape [1, 1, 1, 1] (scalar tensor) with the index of 
        // the logit with the highest value.

        // Reshape to 2D and so we can use get_rows.
        struct ggml_tensor * logits_2d = ggml_reshape_2d(ctx, data->logits, 1, data->logits->ne[0]);
        // shape: {1, 32000, 1, 1}
        struct ggml_tensor * max_logit = ggml_get_rows(ctx, logits_2d, max_idx);
        // This will have shape [1, 1, 1, 1] (scalar tensor) with the value of the max logit (not the index).

        // Subtract the max_logit from all logits.
        struct ggml_tensor * diff = ggml_sub(ctx, data->logits, max_logit);
        // [1.3   0.1  4.5   1.2]
        // [4.5   4.5  4.5   4.5]
        //
        // [-3.2 -4.4  0.0  -3.3]

        // Add small epsilon to make max position strictly positive
        struct ggml_tensor * diff_eps = ggml_scale_bias(ctx, diff, 1.0f, 1e-6f);
        // diff_eps = 1 * diff + 1e-6
        // [(1 * -3.2 + 1e-6) (1 * -4.4 + 1e-6)  (1 * 0.0 + 1e-6)  (1 * -3.3 + 1e-6)]
        // [-3.199999             -4.399999          0.000001         -3.299999]

        // Create the mask for the max logit. step returns 0 for negative values, 1 for positive values.
        struct ggml_tensor * mask = ggml_step(ctx, diff_offset);
        // [-3.199999  -4.399999  0.000001  -3.299999]
        // [ 0              0        1           0   ]

        const float large_val = 1e9f;
        struct ggml_tensor * bias = ggml_scale_bias(ctx, mask, large_val, -large_val);
        // bias = large_val * diff + (-large_val)
        // [ (1e9 * 0) + (-1e9)   (1e9 * 0) + (-1e9)   (1e9 * 1) + (-1e9)   (1e9 * 0) + (-1e9) ]
        // [       -1e9               -1e9                   0                   -1e9          ]

        data->logits = ggml_add(ctx, data->logits, bias);
        // [1.3   0.1  4.5   1.2]
        // [-1e9 -1e9   0   -1e9]
        // [-999999998.7  -999999999.9   4.5   -999999998.8]
        ggml_build_forward_expand(gf, data->logits);
        return;
    }
```
Or much simpler:
```c++
        // Find the most probable token index.
        struct ggml_tensor * max_idx = ggml_argmax(ctx, data->logits);
        ggml_set_name(max_idx, "temp_max_idx");

        data->candidates = max_idx;

        struct ggml_tensor * logit = ggml_reshape_2d(ctx, data->logits, 1, data->logits->ne[0]);
        data->logits = ggml_get_rows(ctx, logit, max_idx);
```
