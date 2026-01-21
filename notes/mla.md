## Multi-Head Latent Attention (MLA)
This is an attention mechanism that was introduced by DeepSeek (V2) and its goal
is to reduce the massive memory usage of the KV-cache during long context
inference.

In multi-head attention the key and value matrix must be stored for every token
in the conversation history. Grouped query attention (GQA) tries to reduce this
memory by sharing key and value matrices across multiple query heads. But even
with GQA as the context grows larger and larger the KV-cache can still grow to
hundreds of gigabytes in size, which can make it impossible to run on most
consumer devices. This is where MLA comes in.

The idea is that instead of storing the full matrices for the keys and values
they are projected down into a smaller latent space (hence latent in the name).
So when the model processes a token embedding it projects the Key and Value into
a low-dimensional latent vector using a down projection. And this vector is what
is stored.

In a Standard Transformer (MHA), the Attention block works like this:
1. Input: Takes the vector from the previous layer (h).
2. Expand: Multiplies h by three giant matrices (W_Q, W_K, W_V) to produce three
   huge vectors: Query (Q), Key (K), and Value (V).
3. Cache: Stores the huge K and V vectors (after the have been projected by the
   previous step, in the KV Cache.
4. Attend: Calculates attention scores using Q and the cached K.

In MLA the Attention block works like this:
1. Input: Takes the vector from the previous layer (h).
2. Compress (Down-Project): Multiplies h by a small "Compression Matrix" (W_Down)
   to create a small latent vector (c_KV).
3. Cache: Stores only this small c_KV in the KV Cache (and rope information, but more on this later).
4. Expand (Up-Project): When we need to calculate scores, the model mathematically
   "unpacks" c_KV back into usable Keys and Values (conceptually) to do the math.

Having to compress using a down project matrix multiplication and then later
expand using an up project matrix multiplication does add some computational
overhead. But it was shown that the uncompressing can actually be avoided.  If
we imagine we have:
```console
Query * (W_unzip* compressed_key)
```
Since multiplication is associative we can rearrange this to:
```console
q_absorbed = (Query * W_unzip) * compressed_key
```
And this means the same thing. And this called "absorbing" the unzip_matrix. We
can do this as the W_unzip is fixed and known ahead of time.
Because Q_absorbed already contains the "W_unzip" matrix inside it, multiplying
it against the compressed key mathematically expands the key on the fly.

But we also have to account for the position encoding like RoPE, which tells the
model that "Dan" came before "loves".

First we have two matrices which are `W_down` which is used to compress, and
`W_unzip` which is used to decompress. And the matrix absorption trick happens
during the read step where which we saw earlier.

For RoPE we would have to:
* Rotate the key based on its position R_pos.
* Then compress the rotated key using the down projection.
```console
score = Query * (W_down * (R_pos * Key))
```
Now, like we mentioned earlier we want to use the absorption trick to avoid and
that was possible because we could merge the W_unzip matrix into the Query matrix
and then perform the multiplication with the compressed key. But now we have
the rope information in R_pos which is position dependent, that is is it different
for each token embedding in the sequence. So before we could use Query * W_unzip
and then multiply by the compressed key because W_unzip was fixed. But now with
R_pos it would mean that we would have to include this position information in
the multiplication which would then have to be done for every token in the
sequence which would defeat the purpose of the whole exercise.
For example, if we are currently processing the 3rd token in the sequence we would
need to look back at the 2nd token which means we would need to rotate the Query
vector based on the position of the 2nd token. And this would have to be
done for every token in the sequence which would be very expensive.
```console
Absorbed_Query = Query * W_Unzip * R_pos
```
We would not be able to create a single absorbed query matrix that works for all
tokens because R_pos is different for each token position. So instead we would
need one absorbed query matrix per token position which would be very expensive.
We would be recreating the the matrix for every token in the sequence which is
slower than just unzipping the key directly.

Could we not just RoPE before compressing the key then?  
No, because recall that RoPE relies on pairs and rotates those pairs of values
based on their position. This precise gemetric relationship is crucial for RoPE
to encode the distance. Compressing this is like smearing all the dimensions
togeher and losing the precise positional relationships between the pairs.

The solution to both of these issue is to use a "side car" for the RoPE
information. This allows us to keep the positional information separate from the
content information and we can still use the absorption trick for the content.
And it also preserves the positional information. We have to then store the
rope information in the kv-cache along with the compressed content.

So the cache in memory might looks like this for each token:
```console
[compressed content (512 floats)] + [pos info (64 floats)]
```

So we have two tracks/paths:
```console
Hidden State (h)
     |
     |--- Path 1: Multiply by W_Down_Content -> [Compressed Content] (Big part)
     |
     |--- Path 2: Multiply by W_Down_Rope    -> [Raw Position]       (Tiny part)
```

No-rope:
1. The compressed content path (what we are talking about)
   q_content (absorbed/projected)
   k_content (from the compressed cache C_KV)
   score_content = q_content . k_content

Rope:
2. The uncompressed positional info path (where in the sentence/sequence we are)
    q_rope (a small vector)
    k_rope (a small vector)
    score_rope = q_rope . k_rope

The final attention score is then:
```console
Final Score = score_content + Score_rope
```
So the KV-Cache in DeepSeek-V3 actually stores two things for every token:
1. The compressed content vector (C_KV)
2. The RoPE Key (k_rope) not compressed


### DeepSeek V2 llama.cpp walkthrough
```c++
llm_build_deepseek2::llm_build_deepseek2(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {
    // lite variants include DeepSeek-V2-Lite, GigaChat3-10B-A1.8B
    bool is_lite = (hparams.n_layer == 27 || hparams.n_layer == 26);

    const bool is_mla = (hparams.n_embd_head_k_mla != 0 && hparams.n_embd_head_v_mla != 0);

    // note: these are the actual head sizes you get when treating as MHA or after "decompression" using wv_b for MLA
    const int64_t n_embd_head_k = is_mla ? hparams.n_embd_head_k_mla : hparams.n_embd_head_k;
    const int64_t n_embd_head_v = is_mla ? hparams.n_embd_head_v_mla : hparams.n_embd_head_v;

    const int64_t n_embd_head_qk_rope = hparams.n_rot;
    const int64_t n_embd_head_qk_nope = n_embd_head_k - n_embd_head_qk_rope;
```
```console
(gdb) p n_embd_head_k
$3 = 192
(gdb) p n_embd_head_v
$4 = 128
```

So like we discusesed earlier we will have one path which handles the rope
and one that does not which is the nope below, which stands for no position
encoding:
```console
(gdb) p n_embd_head_qk_rope
$1 = 64
(gdb) p n_embd_head_qk_nope
$2 = 128
```
The rope sidecare which is stored non-compressed has a dimension of 64.

Next we build the input embeddings, kv cache input, outputs, etc like we normall
would for most transformer model graphs.
Then we will iterate over all the layers:
```c++
    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;
```
And just to see the shape of the input to the attention layer:
```console
(gdb) p cur->ne
$9 = {2048, 1, 1, 1}
```
And for each layer:
```c++
        // self_attention
        {
            ggml_tensor * q = NULL;
            if (!is_lite) {
                ...
            } else {
                q = ggml_mul_mat(ctx0, model.layers[il].wq, cur);
                cb(q, "q", il);
            }
```
So first we have we create the query tensor operation using the weight matrix
from for this layer from the model.
```console
(gdb) p model.layers[0].wq->ne
$38 = {2048, 3072, 1, 1}
```
The weight matrix Wq actually contains two concatenated matrices, one for W_nope
and one for W_rope (the fused query projection matrix). This is save from having
to do two separate matrix multiplications.

```console
(gdb) p q->ne
$15 = {3072, 1, 1, 1}
```
Next, we will create a view into this the query tensor which will hold/view
the non-rope part of the query:
```c++
            // split into {n_embd_head_qk_nope, n_head, n_tokens}
            ggml_tensor * q_nope =
                ggml_view_3d(ctx0, q, n_embd_head_qk_nope, n_head, n_tokens, ggml_row_size(q->type, n_embd_head_k),
                             ggml_row_size(q->type, n_embd_head_k) * n_head, 0);
            cb(q_nope, "q_nope", il);
```
So this is the no rope (nope) query tensor which has shape:
```console
(gdb) p q_nope->ne
$16 = {128, 16, 1, 1}
```

Next we have the query tensor view that will be rope encoded:
```c++

            // and {n_embd_head_qk_rope, n_head, n_tokens}
            ggml_tensor * q_pe = ggml_view_3d(
                ctx0, q, n_embd_head_qk_rope, n_head, n_tokens, ggml_row_size(q->type, n_embd_head_k),
                ggml_row_size(q->type, n_embd_head_k) * n_head, ggml_row_size(q->type, n_embd_head_qk_nope));
            cb(q_pe, "q_pe", il);
```
And the shape of the position encoded query tensor is:
```console
(gdb) p q_pe->ne
$17 = {64, 16, 1, 1}
```
Now in the paper we might have read that we project to q_nope = x * W_nope, and
Q_rope = x * W_rope. But notice that is multiplying the input x twice. What we
are doing above is multiplying once which means that the weight matrix Wq is
actually a concatenation of W_nope and W_rope:

Next we have the KV compressed + position embedding:
```c++
            ggml_tensor * kv_cmpr_pe = ggml_mul_mat(ctx0, model.layers[il].wkv_a_mqa, cur);
            cb(kv_cmpr_pe, "kv_cmpr_pe", il);
```
This layers weight is named `kv` because it has a submatrix which contains the
weights for the keys and values. `a` in LoRA terms stands for A, down/compression
projection (B would be an up projection). `mqa` stands for multi-query attention
even though we are using mla we are compressing everything into a single latent
vector shared by all heads so it is like MQA with 1 head.
So just like the query matrix this is a fused matrix that produces a tensor
that contains both the compressed content (512 dimentions) and the rope information
(64 dimensions) for a total of 576 dimensions.
```console
(gdb) p model.layers[0].wkv_a_mqa->ne
$43 = {2048, 576, 1, 1}

(gdb) p kv_cmpr_pe->ne
$45 = {576, 1, 1, 1}
```

Then we create a view into the kv_cmpr_pe tensor which holds the compressed
content:
```c++
            // split into {kv_lora_rank, n_tokens}
            ggml_tensor * kv_cmpr =
                ggml_view_2d(ctx0, kv_cmpr_pe, kv_lora_rank, n_tokens,
                             ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope), 0);
            cb(kv_cmpr, "kv_cmpr", il);
```
```console
(gdb) p kv_cmpr->ne
$46 = {512, 1, 1, 1}
```

```c++
            // and {n_embd_head_qk_rope, 1, n_tokens}
            ggml_tensor * k_pe = ggml_view_3d(ctx0, kv_cmpr_pe, n_embd_head_qk_rope, 1, n_tokens,
                                              ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
                                              ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
                                              ggml_row_size(kv_cmpr_pe->type, kv_lora_rank));
            cb(k_pe, "k_pe", il);
```
```console
(gdb) p k_pe->ne
$47 = {64, 1, 1, 1}
```

```c++
            q_pe = ggml_rope_ext(ctx0, q_pe, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                 ext_factor, attn_factor, beta_fast, beta_slow);
            cb(q_pe, "q_pe", il);
```
```c++

            k_pe = ggml_rope_ext(ctx0, k_pe, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                 ext_factor, attn_factor, beta_fast, beta_slow);
            cb(k_pe, "k_pe", il);
```
```c++

            kv_cmpr = build_norm(kv_cmpr, model.layers[il].attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);
            cb(kv_cmpr, "kv_cmpr", il);

            if (is_mla) {
                // {n_embd_head_qk_nope, n_tokens, n_head}
                q_nope = ggml_permute(ctx0, q_nope, 0, 2, 1, 3);
                cb(q_nope, "q_nope_perm", il);

                // {n_embd_head_qk_nope, kv_lora_rank, n_head} x {n_embd_head_qk_nope, n_tokens, n_head}
                ggml_tensor * q_nope_absorbed = ggml_mul_mat(ctx0, model.layers[il].wk_b, q_nope);
                cb(q_nope_absorbed, "q_nope_absorbed", il);

                // {kv_lora_rank, n_head, n_tokens}
                q_nope_absorbed = ggml_permute(ctx0, q_nope_absorbed, 0, 2, 1, 3);
                cb(q_nope_absorbed, "q_nope_absorbed_perm", il);

                // {n_embd_head_qk_rope + kv_lora_rank, n_head, n_tokens}
                // note: rope must go first for in-place context shifting in build_rope_shift()
                ggml_tensor * Qcur = ggml_concat(ctx0, q_pe, q_nope_absorbed, 0);
                cb(Qcur, "Qcur", il);

                kv_cmpr = ggml_reshape_3d(ctx0, kv_cmpr, kv_lora_rank, 1, n_tokens);
                cb(kv_cmpr, "kv_cmpr_reshape", il);

                // {n_embd_head_qk_rope + kv_lora_rank, 1, n_tokens}
                ggml_tensor * Kcur = ggml_concat(ctx0, k_pe, kv_cmpr, 0);
                cb(Kcur, "Kcur", il);

                // {kv_lora_rank, 1, n_tokens}
                ggml_tensor * Vcur = kv_cmpr;
                cb(Vcur, "Vcur", il);

                if (inp_attn_scale) {
                    // apply llama 4 temperature scaling
                    Qcur = ggml_mul(ctx0, Qcur, inp_attn_scale);
                    cb(Qcur, "Qcur_attn_temp_scaled", il);
                }

                // note: MLA with the absorption optimzation converts into MQA (ie: GQA with 1 group)
                cur = build_attn(inp_attn,
                        model.layers[il].wo, NULL,
                        Qcur, Kcur, Vcur, nullptr, nullptr, model.layers[il].wv_b, kq_scale, il);

```
