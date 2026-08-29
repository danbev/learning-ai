### Per-Layer Embeddings (PLE)
This is a feature that enables lookup of recalled facts/information by using
a lookup table instead of processing layers in the model. This works by taking
bigrams, and trigrams from the sequence and 


### qwen4exp `set_input`

```c++
void llm_graph_input_ple::set_input(const llama_ubatch * ubatch) {
    const auto & hp = pmodel.hparams;

    const llama_token img_tok = hp.ple_image_token_id != 0
        ? (llama_token) hp.ple_image_token_id
        : (llama_token) hp.ple_eos_token_id;

    auto tok_of = [&](int64_t k) -> llama_token {
        return ubatch->token ? ubatch->token[k] : img_tok;
    };

    const int64_t n_tokens = ubatch->n_tokens;
    const int64_t n_gram   = hp.ple_ngram_size;
    const int64_t n_heads  = hp.ple_n_heads;
    const int64_t per_gram = hp.ple_heads_per_ngram;
    const int64_t eos      = hp.ple_eos_token_id;
    const int64_t n_prev   = n_gram - 1;

    std::vector<int32_t> idx(n_heads * n_tokens);

    GGML_ASSERT(mctx != nullptr);

    mctx->get_prev_tokens(*ubatch, n_prev, prev);
```
Before we step through the rest here are the values for this session:
```console
(gdb) p n_tokens
$35 = 42

(gdb) p n_gram
$36 = 3

(gdb) p n_heads
$39 = 16

(gdb) p per_gram
$37 = 8

(gdb) p n_prev
$38 = 2

(gdb) p hp.ple_layer_multipliers.size()
$40 = 8

(gdb) p hp.ple_layer_multipliers
$46 = {_M_elems = {23703573157769, 20109073645365, 8052911324071, 0, 0, 0, 0, 0}}

(gdb) p hp.ple_head_vocab_sizes.size()
$41 = 64

(gdb) p hp.ple_head_offsets.size()
$42 = 64
```

So this following will loop over all the tokens in the ubatch:
```c++
    for (int64_t i = 0; i < n_tokens; ++i) {
        // ctx is the local context used to construct the n-grams.
        std::vector<int64_t> ctx(n_gram);
        ctx[0] = tok_of(i);
        // ctx[0] = current token id
        // ctx[1] = token id one position back
        // ctx[2] = token id two positions back

        bool cut = false;

        // n_gram = 3
        for (int64_t s = 1; s < n_gram; ++s) {
            // predecessor s positions back; prev[] is oldest-first, missing entries are LLAMA_TOKEN_NULL
            // So prev will contain [two positions back, one position back]
            // n_prev = 2
            // So the first iteration this will become:
            // i * n_prev + (n_prev - s)
            // 0 * 2      + (2      - 1) = 1  prev[1] = token id one position back
            // 0 * 2      + (2      - 2) = 0  prev[0] = token id two positions back
            const llama_token t = cut ? LLAMA_TOKEN_NULL : prev[i*n_prev + (n_prev - s)];

            cut = cut || t < 0 || t == eos;
            // and notice that s is initially 1 so we don't overwrite the first
            // token id (the current token id)
            ctx[s] = cut ? eos : t;
        }

        // n_gram = 3. So we will have two iterations. The first iteration will
        // add indices for the bigrams, and the second the trigrams.
        for (int64_t n = 2; n <= n_gram; ++n) {

            // So first we multiple the current token id with a position multiplier
            // We do this to avoid an ordering issue when later using XOR as it
            // does not take order into account. Or rather the order does not matter
            // for xor but for use it is important we preserve order of token ids
            // or "not good" and "good not" would xor to the same value.
            uint64_t mixed = (uint64_t) ctx[0] * hp.ple_layer_multipliers[0];

            // The following loop will handle both bigram and trigram, notice
            // we are using n in this loop which will be 2 for bigrams but 3
            // for trigrams.
            for (int64_t j = 1; j < n; ++j) {
                // where we xor with mixed (which is the first token id times the
                // first position multiplier.
                mixed ^= (uint64_t) ctx[j] * hp.ple_layer_multipliers[j];
            }
            // mixed is now are hash for this pair.

            // n = 2, per_gram = 8, so base will be 0 in the first iteration
            const int64_t base = (n - 2) * per_gram;

            // here we are going to loop over all per_gram (8)
            for (int64_t g = 0; g < per_gram; ++g) {
                // h_i will be [0, 1, 2, 3, 4, 5, 6, 7]
                const int64_t h_i = base + g;

                // n_heads = 16. 
                // idx[i * 16 + 0 ... i*16 +  7] = 8 bigram  row indices
                // idx[i * 16 + 8 ... i*16 + 15] = 8 trigram row indices
                idx[i * n_heads + h_i] =
                    (int32_t) (mixed % hp.ple_head_vocab_sizes[h_i] + hp.ple_head_offsets[h_i]);
            }
        }
    }

    ggml_backend_tensor_set(rows, idx.data(), 0, idx.size()*ggml_element_size(rows));
}
```
```console
(gdb) p this.mctx.lctx->model->per_layer_tok_embd->ne
$85 = {160, 320001536, 1, 1}

0           [0               159]
            [0               159]
            .
            .
            .
            .
            .
            .
320001535   [0               159]
```
This is just one large tensor but it is divided into logical heads as follows:
```
head 0:
0           [0               159]
            .
            .
            .
20000002    [0               159]


head 1:
20000003    [0               159]
            .
            .
            .
40000025    [0               159]


head 2:
40000026    [0               159]
            .
            .
            .
60000059    [0               159]
```
Notice that each offset is the sum of the previous tables sizes:
```console
offset[1] = offset[0] + vocab_size[0]
```
So `mixed % hp.ple_head_vocab_sizes[h_i]` might produce:
```console
(gdb) p mixed % hp.ple_head_vocab_sizes[0]
$98 = 13981229
```
So what is this? It is a head local row number, that is a row number in one of
the above heads, specifically `head_0` so not a great example perpahs as it is
also a row index into the tensor. But for other heads this will be a row into
a head and we get to that row by adding the offset.

```console
(gdb) p ctx[0]
$70 = 248045

(gdb) p this.mctx.lctx->model->vocab->pimpl->id_to_token[ctx[0]]
$69 = {text = "<|im_start|>", score = 0, attr = LLAMA_TOKEN_ATTR_CONTROL}

(gdb) p ctx[j]
$71 = 248044
(gdb) p this.mctx.lctx->model->vocab->pimpl->id_to_token[ctx[j]]
$72 = {text = "<|endoftext|>", score = 0, attr = LLAMA_TOKEN_ATTR_CONTROL}
(gdb) p j
$73 = 1

(gdb) p *hp.ple_head_offsets._M_elems@16
$19 = {0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297, 160000374, 180000455, 200000548,
  220000655, 240000802, 260000955, 280001114, 300001275}

```

So after this the rows tensor in backend will have been filled with row
indices into the `per_layer_embd` tensor. For each token in the ubatch there will
be 16 indices, 8 for bigrams and 8 for trigrams. These indices will be used with
`ggml_get_rows` to get those hidden vector embeddings from the `per_layer_embd`
tensor.

So just to recap or get my barings a little. When the qwen4exp graph is built
which actually happens before `set_input` so I should probably reorder this
document when I've finished this walk through, we will see the function
`build_inp_ple` being called:
```c++
llama_model_qwen4exp::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_build_delta_net_base(params), model(model) {
    ...

    const auto * mctx_hyb = static_cast<const llama_memory_hybrid_idx_context *>(inp->mctx);
    ...

    ggml_tensor * ple_emb = nullptr;
    if (hparams.ple_n_heads > 0) {
        ple_emb = build_inp_ple(mctx_hyb);
        // make sure ple_emb and build_inp_embd are in the same graph split
        ggml_build_forward_expand(gf, ple_emb);
    }
```
```c++
ggml_tensor * llama_model_qwen4exp::graph::build_inp_ple(
        const llama_memory_hybrid_idx_context * mctx_hyb) {
    const int64_t n_heads = hparams.ple_n_heads;

    auto ple_inp = std::make_unique<llm_graph_input_ple>(
            static_cast<const llama_model_qwen4exp &>(model), mctx_hyb->get_attn());

    ple_inp->rows = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_heads * n_tokens);
    ggml_set_input(ple_inp->rows);
    ggml_tensor * rows = ple_inp->rows;
    res->add_input(std::move(ple_inp));

    // gather then flatten the heads: get_rows lays the head dimension out slowest, as the reference does
    ggml_tensor * emb = ggml_get_rows(ctx0, model.per_layer_tok_embd, rows);
    emb = ggml_reshape_2d(ctx0, emb, hparams.ple_head_dim * n_heads, n_tokens);
    cb(emb, "ple_embd", -1);

    return emb;
}
```
Notice here that rows `ple_embd` is a tensor.
```c++
void llama_model_qwen4exp::load_arch_tensors(llama_model_loader & ml) {
    ...
    if (hparams.ple_n_heads > 0) {
        const std::string ple_name = tn(LLM_TENSOR_PER_LAYER_TOKEN_EMBD, "weight").str();
        const auto & ple_w = ml.require_weight(ple_name.c_str());
        const int64_t ple_rows = ple_w.tensor->ne[1];
        ...

        per_layer_tok_embd = create_tensor(tn(LLM_TENSOR_PER_LAYER_TOKEN_EMBD, "weight"),
                                           { hparams.ple_head_dim, ple_rows }, TENSOR_READ_LAZY);
    }
```
```console
(gdb) p ple_name
$1 = "per_layer_token_embd.weight"

(gdb) p ple_rows
$2 = 320001536
```

And notice that this tensor is created as `TENSOR_READ_LAZY` and because this
weights buffer isn't GPU compatible ggml_get_rows(model.per_layer_tok_embd, rows)
in `build_inp_ple` that we saw above can only be scheduled on the CPU backend.
And notice that `ple_emd` is added to the graph after so that this is not later
interleaved with and backend operations which could/would cause a graph split. The
initial implementation actually did this was updated to the current code above.

Alright, so lets take a look the start of the layers graph building, where we
first have the hyper residual connections (low rank?):
```c++
    ggml_tensor * res_hc = ggml_repeat_4d(ctx0,
            ggml_reshape_3d(ctx0, inpL, n_embd, 1, n_tokens),
            n_embd, hc, n_tokens, 1);
    cb(res_hc, "hc_init", -1);
```
These simply start out as a copy of the input token embeddings.

Next have the iteration over layers:
```c+

    for (int il = 0; il < n_layer; ++il) {
        res->t_layer_inp[il] = res_hc;

        if (hparams.is_ple(il)) {
            res_hc = build_ple(inp->get_recr(), ple_emb, res_hc, il);
        }
```
And notice that the input to the layer is stored which might be used later for
MTP (but more on that later in a separate section). And then for the configured
PLE layers which is only one in this model, from config.json:
```console
        "ple_layer_ids": [
            2
        ],
```
We will call `build_ple` and we are passing in `ple_emb` tensor and also the 
residual hyper connections tensor:
```c++
ggml_tensor * llama_model_qwen4exp::graph::build_ple(
        llm_graph_input_rs * inp,
        ggml_tensor *        emb,
        ggml_tensor *        hidden,
        int                  il) {
    const int64_t hc      = hparams.dsv4_hc_mult;
    const int64_t hc_dim  = hc * n_embd;

    ggml_tensor * key   = build_lora_mm(model.layers[il].ple_key,   emb);
    ggml_tensor * value = build_lora_mm(model.layers[il].ple_value, emb);
```
So at this point we are only building the graph operations but later when the
graph executeds `emb/ple_emb` will have the looked up bigram/trigram embedding
vectors for the tokens in the current ubatch. So emb is just:
```console
(gdb) p emb->ne
$1 = {2560, 1, 1, 1}
```
This looked up vector embedding for a single token. `ple_key` is a learned
matrix takes embd and linearly reproject it into a different subspace.
```console
(gdb) p model.layers[il].ple_key->ne
$5 = {2560, 10240, 1, 1}
(gdb) p n_embd
$6 = 2560
(gdb) p n_embd * 4   (4 hyper connection streams)
$7 = 10240
```

This size comes from concatenating 16 gathered hash head vectors (160):
```console
16 heads * 160 values = 2560
```
So the key will have the shape:
```console
(gdb) p key->ne
$8 = {10240, 1, 1, 1}
```
And what we want to do here is to figure out how much of the information that
we got from the bigrams/trigrams lookup that we should incoporate into `res_hc`
, the current per-stream residual state). So we want to compare this key
information with the current state by using calculating the dot product between
the key information and the current stream states in `res_hc`
```c++
    ggml_tensor * s = ggml_sum_rows(ctx0, ggml_mul(ctx0, key, query));
    s = ggml_scale(ctx0, s, 1.0f / sqrtf((float) n_embd));

    ggml_tensor * mag  = ggml_sqrt(ctx0, ggml_clamp(ctx0, ggml_abs(ctx0, s), 1e-6f, INFINITY));
    ggml_tensor * gate = ggml_sigmoid(ctx0, ggml_mul(ctx0, ggml_sgn(ctx0, s), mag));
    cb(gate, "ple_gate", il);

    // [n_embd, 1, T] value broadcast across the hc streams, scaled by the gate
    ggml_tensor * v3 = ggml_reshape_3d(ctx0, value, n_embd, 1, n_tokens);
    v3 = ggml_repeat_4d(ctx0, v3, n_embd, hc, n_tokens, 1);

    ggml_tensor * gated = ggml_mul(ctx0, v3, gate);
    cb(gated, "ple_gated_value", il);
```
Value is just the projected embedding vector informations, and the same value is
used for all 4 streams, but that all have their own gate which was calculated
above. This allows each stream to decide how much of this embedding to actually
accept. If a stream's current state strongly agrees with the facts in the proposed
key the gate will have a value near 1, so it would get fully injected, and a
stream that does not will have a gate close to 0 which would be mostly ignored.

The states will be normalized since they have been updated:
```c++
    ggml_tensor * normalized = grouped_norm(
            ggml_reshape_2d(ctx0, gated, hc_dim, n_tokens),
            model.layers[il].ple_norm_conv);
    normalized = ggml_reshape_2d(ctx0, normalized, hc_dim, n_tokens);
```
Now, the next thing to happen is a convolution...
```c++
    const int64_t kern = hparams.ple_conv_kernel;
    const int64_t dil  = hparams.ple_ngram_size;
    const int64_t hist = (kern - 1) * dil;

    // the conv history is per sequence, so the input carries the sequence axis too
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;
```

```console
(gdb) p kern
$10 = 4
(gdb) n
/1171	    const int64_t hist = (kern - 1) * dil;
(gdb) p dil
$11 = 3
(gdb) n
-1174	    const int64_t n_seqs       = ubatch.n_seqs;
(gdb) p hist
$12 = 9
```
Now, recall that `normalized` is the normalized gated state which has been
updated. But we have not updated the res_hs tensor, not yet.
```console
(gdb) p normalized->ne
$22 = {10240, 1, 1, 1}
```
So this tensor contains all 4 streams (2560 x 4 = 10240). This will be passed
to `build_conv_state_at` (at layer?):
```c++
    ggml_tensor * padded = build_conv_state_at(inp, inp->mctx->get_p_l(il),
            ggml_reshape_3d(ctx0, normalized, hc_dim, n_seq_tokens, n_seqs),
            hist, hc_dim, il);
```
Notice that this is passing in the memory context's `p_l` tensor which stores
the history for the convolution. So what we are about to do is to get some
additional context from the past tokens, with the updated stream states.
So we have the following values kern=4, dil=3, hist=9. So the convolution will
be looking back 9 positions (which is why we need the persisted memory)

__new_wip__


```console

(gdb) p n_tokens
$40 = 1

(gdb) p model.layers[il].ple_key->ne
$37 = {2560, 10240, 1, 1}

(gdb) p key->ne
$41 = {10240, 1, 1, 1}

```
Now this is actually not strictly related to PLE but has to do with hyper
connections (TODO: link to notes). This produces a separate key vector for every
hyper-connection stream:
```console
  key[:, 0, token] = PLE key for residual stream 0
  key[:, 1, token] = PLE key for residual stream 1
  key[:, 2, token] = PLE key for residual stream 2
  key[:, 3, token] = PLE key for residual stream 3
```


After the build_ple we have:
```console
        ggml_tensor * inject = nullptr;
        ggml_tensor * cur = build_hc_mix(res_hc,
                model.layers[il].hc_attn_norm,
                model.layers[il].hc_attn_down,
                model.layers[il].hc_attn_up,
                model.layers[il].hc_attn_inject,
                &inject, il);
```


_wip_

```console
(gdb) br qwen4exp.cpp:1029 if n==3
```

