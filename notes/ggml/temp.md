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
