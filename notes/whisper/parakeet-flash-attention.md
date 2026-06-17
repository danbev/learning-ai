### Flash Attention in parakeet.cpp
Notes related to attempting to use flash attention in parakeet.cpp.

Simplified example of relative positions:
```console
d_head      = 2
n_time      = 3
window_size = 5
n_head      = 1
```
So `pos` has a total of 10 floats:
```
idx:   0   1   2   3   4   5   6   7   8   9
pos:  a0  a1  b0  b1  c0  c1  d0  d1  e0  e1
     [ r=0 ] [ r=1 ]  [ r=2 ] [ r=3 ] [ r=4 ]
     [off-2] [off-1]  [off=0] [off+1] [off+1]

pos shape: [2, 5, 1]

pos[:, 0, 0] => what a query looks like if it wants to attent to 2 steps in the past.
pos[:, 1, 0] => what a query looks like if it wants to attent to 1 steps in the past.
pos[:, 2, 0] => what a query looks like if it wants to attent to itself.
pos[:, 3, 0] => what a query looks like if it wants to attent to 1 step in the future.
pos[:, 4, 0] => what a query looks like if it wants to attent to 2 steps in the future.
```
Q_v has the shape [2, 3, 1] so this contains a query vector for each time frame
each of two floats and this represents position biases regardless of the actual
content:
```
   +--+--+
   |q0|q1|
   +--+--+
   |r0|r1|
   +--+--+
   |s0|s1|
   +--+--+
```

```console
struct ggml_tensor * rel_pos_scores = ggml_mul_mat(ctx0, pos, Q_v);

   pos            Q_v^T          t0 t1 t2
   +--+--+                      +--+--+--+
   |a0|a1| (-2)                 |  |  |  |
   +--+--+        +--+--+--+    +--+--+--+
   |b0|b1| (-1)   |q0|r0|s0|    |  |  |  |
   +--+--+      . +--+--+--+  = +--+--+--+
   |c0|c1| ( 0)   |q1|r1|s1|    |  |  |  |
   +--+--+        +--+--+--+    +--+--+--+
   |e0|e1| (+1)                 |  |  |  |
   +--+--+                      +--+--+--+
   |d0|d1| (+2)                 |  |  |  |
   +--+--+                      +--+--+--+
```
So the first column of the result is the distance scores for query 0, the first
column is the score for two time frames into the past. The second column -1 etc.

Now the relative positions are per head, even if I only showed 1 head above.

### PR comment
After the shift trick the relative position scores are per head:
```console
(gdb) p rel_pos_scores->ne
$17 = {625, 625, 8, 1}

[n_time, n_time, n_head]
```
There is a different `n_time x n_time` bias matrix for each of the 8 heads.
The mask to `ggml_flash_attn_ext` is designed for ALiBi where the mask is shared
accross all the heads. This is enforce by the CUDA backend:
```c++
static best_fattn_kernel ggml_cuda_get_best_fattn_kernel(const int device, const ggml_tensor * dst) {
    ...
    if (mask && mask->ne[2] != 1) {
        return BEST_FATTN_KERNEL_NONE;
    }
    ...
}
```
My previous attempt was to fold the `rel_pos_scores` into the FA mask which would
mean that `mask->ne[2] ==  8 (n_heads)` which violates the CUDA constraint, so
that would fallback to a CPU implementation. 

My next attempt was to splitting into a FA call per head, each getting its own
`[n_time, n_time, 1]` mask slice and then passing it to `ggml_flash_attn_ext`,
but that meant calling `ggml_flash_attn_ext` for each head means `n_head`` calls
to the kernal. This turned out to be ~7% slower that the the original softmax
based code.

My understanding is that parakeet's relative-position bias is incompatible with
what FA is optimized for right now.


FA's core trick is to never materialize the full n_time x n_time attention-score
matrix, it streams Q, K, V through fused tiles and produces the output directly,
skipping the writeback/readback of the intermediate score matrix and the softmax
output (probs).

But rel_pos_scores (per head, [n_time, n_time, n_head]) is that full
n_time x n_time matrix, already materialized via the shift trick, before FA ever
runs. There's no way to hand FA a "formula" for this bias the way ALiBi hands it
a simple slope * (j - i) that the kernel can evaluate per-tile on the fly, it's
arbitrary learned content, not a closed form. So the O(n_time^2 * n_head) cost
FA exists to avoid has already been paid by the time we get to ggml_flash_attn_ext.

Given that, what FA can still save is just the writeback/readback of attn_scores
(= content_scores + rel_pos_scores, scaled and masked) and probs (after softmax)
two O(n_time^2 * n_head) passes. That's a real but modest saving.

Against that saving, the per-head split needed to satisfy CUDA's mask->ne[2] == 1
constraint replaces one batched mul_mat/soft_max/mul_mat per layer with n_head
separate FA kernel launches per layer (8x more launches, 192 total across 24
layers). For these sequence lengths, that per-launch overhead outweighs the
savings, hence the ~7% regression.




So the relative positions, they are computed for the position frequencies which
is an input tensor to this graph:
```c++
    struct ggml_tensor * rel_positions = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, window_size);
    ggml_set_name(rel_positions, "rel_positions");
    ggml_set_input(rel_positions);

    struct ggml_tensor * freqs = ggml_repeat_4d(ctx0, pos_freqs, d_half, window_size, 1, 1);
    struct ggml_tensor * theta = ggml_mul(ctx0, freqs, rel_positions);

    struct ggml_tensor * sin_t = ggml_reshape_3d(ctx0, ggml_sin(ctx0, theta), 1, d_half, window_size);
    struct ggml_tensor * cos_t = ggml_reshape_3d(ctx0, ggml_cos(ctx0, theta), 1, d_half, window_size);
    // [n_state, window_size]
    struct ggml_tensor * pos_emb = ggml_reshape_2d(ctx0,
        ggml_cont(ctx0, ggml_concat(ctx0, sin_t, cos_t, 0)), n_state, window_size);
```

And this then multiplied by the layers attention position weights which encodes
"in general, regardless of content, how much to I care about something N steps
away":
```c++
            struct ggml_tensor * pos = ggml_mul_mat(ctx0, layer.attn_pos_w, pos_emb);
```
This is then reshaped for multihead attention:
```c++
            pos = ggml_reshape_3d(ctx0, pos, d_head, n_head, window_size);
            pos = ggml_cont(ctx0, ggml_permute(ctx0, pos, 0, 2, 1, 3));
```
```console
Before reshape:
(gdb) p pos->ne
$13 = {1024, 1249, 1, 1}

After reshape:
(gdb) p pos->ne
$14 = {128, 1249, 8, 1}
```

This tensor is then used a little later:
```c++
                struct ggml_tensor * rel_pos_scores = ggml_mul_mat(ctx0, pos, Q_v);
```
Q_v — the query, with the distance-bias added.

```console
(gdb) p Q_v->ne
$7 = {128, 625, 8, 1}
```
So we have 8 heads in this case, each containing a [128 625] matrix. So 625
time frames with 128 features for each time frame.
And we saw that we also reshaped out position tensor above for multiple attention
heads:
```console
(gdb) p pos->ne
$8 = {128, 1249, 8, 1}
```

```console
(gdb) p rel_pos_scores->ne
$15 = {1249, 625, 8, 1}
        ↑     ↑
      window time
      size   frames
```
So what does this actually represent?  
The window size is 1249 because each time frame can look 625 frame backwards
and 625 frames forward (2*625-1=1249). So this contains the set of possible
relative offset between two frames, -(n_time-1) to +(n_time-1), so -624 ... +624.

So rel_pos_scores is a vector of 1249 numbers — one score per possible distance,
purely about distance, independent of what's actually stored at that key position.


```c++
    // flash_attn_ext only takes a single additive mask, so we fold the
    // Transformer-XL relative position bias into the attention mask:
    // combined_mask = scale*rel_pos_scores + attn_mask, matching the
    // kernel's s = s*scale; s += mask; formula.

    const float kq_scale = 1.0f / std::sqrt((float) d_head);
    struct ggml_tensor * combined_mask = ggml_add(ctx0,
            ggml_scale(ctx0, rel_pos_scores, kq_scale), attn_mask);
    combined_mask = ggml_cast(ctx0, combined_mask, GGML_TYPE_F16);
    ggml_format_name(combined_mask, "enc_%d_attn_combined_mask", il);
```

```console
I applied the flash-ext stash and dug into why it wasn't faster — then fixed the actual bug and re-measured.
Findings:

1. d_head=128 is fine. CUDA's flash attention supports head dim 128 — that wasn't the issue.

2. The original port was silently falling back to CPU — but because of the mask, not the head size.

The Conformer's Transformer-XL relative position bias (rel_pos_scores) is per-head:
shape [n_time, n_time, n_head].

The stash folded this into the FA mask (combined_mask = scale*rel_pos_scores + attn_mask),
giving a mask with ne[2] == n_head == 8. But ggml_cuda_get_best_fattn_kernel()
requires mask->ne[2] == 1 (CUDA FA only supports a mask shared across heads, e.g.
ALiBi-style). So GGML_OP_FLASH_ATTN_EXT was unsupported on CUDA for this graph,
and the scheduler silently routed all 24 enc_X_attn_flash nodes to CPU — confirmed
via GGML_SCHED_DEBUG=2. That forces Q/K/V and a 5MB F16 mask to bounce CUDA↔CPU
every layer.

- Baseline (no FA)               : ~212 ms encode (jfk.wav)
- Original FA port (CPU fallback): ~268 ms (+26%)

3. I fixed the fallback by splitting flash_attn_ext into one call per head (each
with a [n_time,n_time,1] mask slice, satisfying ne[2]==1), then concatenating
the 8 outputs back together. Verified via GGML_SCHED_DEBUG that all 192 FA ops
(24 layers × 8 heads) now run on CUDA0, and the transcription is still correct.

- Per-head FA, fully on CUDA     : ~226 ms (+7% vs baseline, still slower)

4. Why FA still doesn't win, even placed correctly:
- The per-head relative-position bias is itself an O(n²·n_head) tensor that must
be computed and read in full regardless — it's not a cheap closed-form bias FA
can generate on the fly (unlike ALiBi). So the "avoid materializing the O(n²)
score matrix" benefit FA normally provides is mostly already spent.
- Satisfying CUDA's mask->ne[2]==1 constraint requires splitting into 8 small
FA kernel launches per layer (192 total vs ~3 big ops/layer before). For these
sequence lengths (n_time ≈ 600 for jfk.wav, capped at 8192 by
PARAKEET_LOCAL_ATTN_THRESHOLD), launch overhead outweighs any bandwidth savings.
- The local-attention path (used for all real long-audio cases, e.g. gb1.wav, n_time=20875)
wasn't touched at all — it would hit the same per-head mask problem plus the
windowed-layout complexity.

Conclusion:
this isn't a missed-optimization bug — the architecture (per-head O(n²) relative
position bias) is fundamentally a poor fit for ggml_flash_attn_ext's mask model.
Even with the CPU-fallback bug fixed, FA is ~7% slower, not faster.

