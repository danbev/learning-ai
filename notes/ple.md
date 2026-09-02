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
The size of emb will depend on the number of tokens in the current sequence(
n_tokens)
```console
(gdb) p n_tokens
$10 = 13

(gdb) p rows->ne
$11 = {208, 1, 1, 1}
```
So we have 13 tokens and each one has 16 indices (8 bigram, 8 trigram).
And this means that emb will have the following shape:
```console
(gdb) p emb->ne
$12 = {160, 208, 1, 1}
  
0   [0         159]   // first tokens first 160 values for its bigram
          .
7   [0         159]   // first tokens last 160 values for its bigram
8   [0         159]   // first tokens first 160 values for its trigram
          .
15  [0         159]   // first tokens last 160 values for its trigram
          .
          .
207 [0         159]
```
And this is then reshaped:
```c++
    emb = ggml_reshape_2d(ctx0, emb, hparams.ple_head_dim * n_heads, n_tokens);
                                     [ 160 * 16 = 2560             ] [  13   ]
```
Which produces:

```console
(gdb) p emb->ne
$15 = {2560, 13, 1, 1}
```
So for each token in this batch we have retrieved a vector embedding for the
current token id and the past bigram and trigram. So this does depend on preceding
tokens so it is not a "look up this token id in total isolation which one might
think when just hearing a simplified description of this. The dependency is on
raw token identity, the literal token ids are hashed together so there is no
learned notion of context (yet). A bit later we will see how this information is
gated with the hyper connection streams which do have context.

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
updated. But we have not updated the `res_hc` tensor (not yet).
```console
(gdb) p normalized->ne
$22 = {10240, 1, 1, 1}
```
So this tensor contains all 4 streams (2560 x 4 = 10240). This will be passed
to `build_conv_state_at`:
```c++
    ggml_tensor * padded = build_conv_state_at(inp, inp->mctx->get_p_l(il),
            ggml_reshape_3d(ctx0, normalized, hc_dim, n_seq_tokens, n_seqs),
            hist, hc_dim, il);
```
Notice that this is passing in the memory context's `p_l` tensor which stores
the history for the convolution. So what we are about to do is to get some
additional context from the past tokens. And we area also passing in the
normalized tensor from above.

So we have the following values kern=4, dil=3, hist=9. So the convolution will
be looking back 9 positions (which is why we need the persisted memory), that is
we will read the current 9 positions from memory and also update the memory (
adding an operation) with the lastest 9 positions. The only positions will be
shifted out, or replaced entirely depending on how many token are in the current
sequence.

So lets take a closer look at `build_conv_state_at`:
```c++
ggml_tensor * llama_model_qwen4exp::graph::build_conv_state_at(
        llm_graph_input_rs * inp,
        ggml_tensor *        conv_states_all,  // p_l
        ggml_tensor *        x,                // reshaped normalized (gated lookuped info)
        int64_t              state_cols,       // hist     (9)
        int64_t              channels,         // hc_dim   (10240)
        int                  il) {
```

Now, x will vary depending on the sequence length, just keep this in mind if you
set a break point here and just run in the debugger, the first time it hits will
be for reserve calls in `llama_context`'s constructor so the sequence length will
be 1.
```console
(gdb) p x->ne
$28 = {10240, 1, 1, 1}

(gdb) p conv_states_all->ne
$29 = {92160, 1, 1, 1}

(gdb) p 10240 * 9
$30 = 92160
```

Next we have:
```c++
    auto it = rs_rows.find(conv_states_all);
    if (it == rs_rows.end()) {
        it = rs_rows.emplace(conv_states_all, build_rs(inp, conv_states_all, row_total, n_seqs)).first;
    }
    ggml_tensor * rows = it->second;
```
This is doing a lookup to see if we have already added a copy operation to the
graph for the tensor pointer `conv_states_all`, which is performed by calling
`build_rs`. If we have, we can just reuse it but otherwise we schedule a write
back operation of the currently inactive rows. The rows tensor looks like this:
```console
(gdb) p rows->ne
$36 = {92160, 1, 1, 1}
```
So this is storing the history for the past 9 tokens. This is reshaped into a
state tensor:
```c++
    ggml_tensor * state = ggml_reshape_3d(ctx0, rows, state_cols, channels, n_seqs);
```
```console
(gdb) p state->ne
$37 = {9, 10240, 1, 1}
```
And the we concatenate the history with x, which recall is our normalized gated
looked up information about (sourced from the bigrams/trigrams in the token sequence):
```c++
    ggml_tensor * conv_input = ggml_concat(ctx0, state, ggml_transpose(ctx0, x), 0);
```
```console
    [ 9 tokens history | gated looked up info]
```
Then we have:
```c++
    // keep the last state_cols columns for the next ubatch
    const size_t row_size = ggml_row_size(conv_states_all->type, row_total);

    ggml_tensor * tail = ggml_view_3d(ctx0, conv_input,
            state_cols, channels, n_seqs,
            conv_input->nb[1], conv_input->nb[2],
            ggml_row_size(conv_input->type, conv_input->ne[0] - state_cols));
```
So we have something like this:
```console
conv_input = {10, 10240, 1, 1}
0     [0     9]
.
.
.
10239 [0     9]
```
And the above code is creating a view into this using ne[0]=9, ne[1]=10240,
ne[2]=1, and the final argument is the offset=1.
```console
(gdb) p ggml_row_size(conv_input->type, conv_input->ne[0] - state_cols)
/$50 = 4
```
So we have offset each row by 4 bytes, and since the type of ne[0] is `GGML_TYPE_F32`
we are skipping one entry initially and then the strides will do the rest and
naturally skip the first element, which on this case is the oldest token in the
history which we are "evicting/filtering" from this view:
```console
0     [1    9]
.
.
.
10239 [1    9]
```
Think of `conv_input` as an array were we start by indexing 4 bytes in. Then the
strides will use that offset.
```console
(gdb) p tail->ne
$53 = {9, 10240, 1, 1}
```

The destination for this is the convolution state, which is what we want to update
and we create a 2d view into it using the following:
```c++
    ggml_tensor * dst = ggml_view_2d(ctx0, conv_states_all,
            state_cols * channels, n_seqs,
            conv_states_all->nb[1],
            kv_head * row_size);
```
So we are creating a tensor for the destination which is the conv_states_all
tensor, followed by the actual copy operation before returning conv_input:
```c++
    ggml_build_forward_expand(gf, ggml_cpy(ctx0, ggml_cont(ctx0, tail), dst));

    return conv_input;
```
So this will make sure that when the graph is executed we will store away the
updated convolution history, bumping out any old history.

So that brings us back to:
```c++
    // [hist + n_seq_tokens, hc_dim, n_seqs], tokens on ne[0]
    ggml_tensor * padded = build_conv_state_at(inp, inp->mctx->get_p_l(il),
            ggml_reshape_3d(ctx0, normalized, hc_dim, n_seq_tokens, n_seqs),
            hist, hc_dim, il);
```
The name padded is referring to that this will be used in the convolution as
padding, providing the history of past tokens for the convolution operation.
For the very first token this will infact be a zero padding as `build_rs` zeros
out a sequences cache row using `ggml_scale_inplace(state_zero, 0)` in llama-graph.cpp

And this will be used in the actual convolution operation below:
```c++
    ggml_tensor * conv_out = nullptr;

    // kern = 4 in our case. So this will loop four times, once for each tap.
    for (int64_t k = 0; k < kern; ++k) {
        // tap k reads (kern-1-k)*dilation positions back, dil = 3
        const int64_t start = hist - (kern - 1 - k) * dil;

        ggml_tensor * shifted = ggml_cont(ctx0,
                ggml_transpose(ctx0,
                        ggml_view_3d(ctx0, padded, n_seq_tokens, hc_dim, n_seqs,
                                padded->nb[1], padded->nb[2],
                                ggml_row_size(padded->type, start))));

        // column k of the [kern, hc_dim] kernel is one weight per channel
        ggml_tensor * wk = ggml_cont(ctx0,
                ggml_view_2d(ctx0, model.layers[il].ple_conv1d, 1, hc_dim,
                        model.layers[il].ple_conv1d->nb[1],
                        k * model.layers[il].ple_conv1d->nb[0]));
        // this kernel keeps the file type, so cast it before it multiplies an f32 activation
        wk = ggml_reshape_1d(ctx0, wk, hc_dim);
        if (wk->type != GGML_TYPE_F32) {
            wk = ggml_cast(ctx0, wk, GGML_TYPE_F32);
        }

        ggml_tensor * term = ggml_mul(ctx0, shifted, wk);
        conv_out = conv_out ? ggml_add(ctx0, conv_out, term) : term;
    }
```
So if we have n_tokens=42, this would mean that padded would be:
```console
(gdb) p padded->ne
$19 = {51, 10240, 1, 1}

(gdb) p model.layers[il].ple_conv1d->ne
$11 = {4, 10240, 1, 1}

(gdb) p shifted->ne
$23 = {10240, 42, 1, 1}

(gdb) p wk->ne
$24 = {10240, 1, 1, 1}
```
```console
         padded tensor

0     [0        50]     channel 0
1     [0        50]     channel 1
2     [0        50]
3     [0        50]
4     [0        50]
5     [0        50]
6     [0        50]
7     [0        50]
8     [0        50]
           .
           .
           .
10239 [0        50]
       ↑        ↑
       t0       t50


           shifted tensor
0     [0                             10239]  token 0
                  .
                  .
                  .
41    [0                             10239]  token 42


           wk tensor
0     [0                             10239]

```
Lets look at the first row in padded, where we have the 9 history tokens first
followed by the 42 new tokens (ple gated tokens):
```console
 0  1  2  3  4  5  6  7  8| 9  10  11  12  13 ... 50
[---------history (9)----]|[t0 t1  t2  t3  t4     t41]
```
So we will be looping over k which is 4 so the local variable `start` will take
on the following values:
```console
        const int64_t start = hist - (kern - 1 - k) * dil;
hist=9
kern=4
dil=3


k=0, start=9 - (4 - 1 - 0) * 3=0: window=cols[0 ... 41]
k=1, start=9 - (4 - 1 - 1) * 3=3: window=cols[3 ... 44]
k=2, start=9 - (4 - 1 - 2) * 3=6: window=cols[6 ... 47]
k=3, start=9 - (4 - 1 - 3) * 3=9: window=cols[9 ... 50]
```
Start is then used to create a view into the padded tensor, and notice that we
are using start to get the byte offset (0, 12, 24, 36):
```console
(gdb) p ggml_view_3d(ctx0, padded, n_seq_tokens, hc_dim, n_seqs, padded->nb[1], padded->nb[2], ggml_row_size(padded->type, start))->ne
$34 = {42, 10240, 1, 1}
```
This is then made contiguous by using `ggml_cont`.
Next we have wk which I guess is the kernel weight for this k (the current tap):
```c++
        ggml_tensor * wk = ggml_cont(ctx0,
                ggml_view_2d(ctx0, model.layers[il].ple_conv1d, 1, hc_dim,
                        model.layers[il].ple_conv1d->nb[1], // 8 bytes stride
                        k * model.layers[il].ple_conv1d->nb[0]));
                        //offset: 0 * 2 = 0
                                  1 * 2 = 2
                                  2 * 2 = 4
                                  3 * 2 = 6
```
```console
(gdb) p model.layers[il].ple_conv1d->ne
$58 = {4, 10240, 1, 1}

0     [0     3]
          .
          .
          .
10239 [0     3]

(gdb) p ggml_view_2d(ctx0, model.layers[il].ple_conv1d, 1, hc_dim, model.layers[il].ple_conv1d->nb[1], k * model.layers[il].ple_conv1d->nb[0])->ne
$57 = {1, 10240, 1, 1}
0    [0]
      .
      .
      .
10239[0]
```
So this is view of a column of `ple_conv1d`, which is also why the offset is
0, 2, 4, 6 (the type of the tensor is `GGML_TYPE_F16` so two bytes per entry.
So the shape of wk is initially [1, 10240, 1, 1] and this is then reshaped
into [10240, 1, 1, 1]:
```c++
        wk = ggml_reshape_1d(ctx0, wk, hc_dim);
```
And there is also a cast to f32 if it is not already:
```c++
        if (wk->type != GGML_TYPE_F32) {
            wk = ggml_cast(ctx0, wk, GGML_TYPE_F32);
        }
```
Next we have the first part of the convolution, and keep in mind that since
we have:
```console
(gdb) p shifted->ne
$63 = {10240, 42, 1, 1}
(gdb) p wk->ne
$64 = {10240, 1, 1, 1}
```
The shapes don't match for second dimension, but in ggml the have to match, or
if wk's size is 1, then ggml just repeats wk's single slice accross the whole
dimension of shifted.

```c++
        ggml_tensor * term = ggml_mul(ctx0, shifted, wk);
        shifted                     wk (broadcasted)
0       [0          10239]       [0          10239] "real row"
1       [0          10239]       [0          10239] broadcasted
2       [0          10239]       [0          10239] broadcasted
.              .                  .
.              .                  .
.              .                  .
41      [0          10239]       [0          10239] broadcasted
```
So the same kernel is applied to each row in shifted.
Now, I'm struggling to actually see how the kernel is applied so lets try a
simplified example:
```console
kern=2,  dil=1, hist=1
n_seq_tokens=4 [x0, x1, x2, x3]

       one history
          ↓
padded = [h0, x0, x1, x2, x3]

kernel: [w0, w1]

Iterations:
k=0 :
shifted_0 = [h0, x0, x1, x2]
shifted_0 * kernel_0
[h0, x0, x1, x2] [w0]  broadcasted  = [h0*w0, x0*w0, x1*w0, x2*w0]
                 [w0]      ↓
                 [w0]      ↓
                 [w0]      ↓

term_0 = [h0*w0, x0*w0, x1*w0, x2*w0]

k=1: 
shifted_1 = [x0, x1, x2, x3]
shifted_1 * kernel_1
[x0, x1, x2, x3] [w1]  broadcasted  = [x0*w1, x1*w1, x2*w1, x3*w1]
                 [w1]      ↓
                 [w1]      ↓
                 [w1]      ↓

term_1 = [x0*w1, x1*w1, x2*w1, x3*w1]

conv_out = term_0 + term_1
         = [h0*w0 + x0*w1, x0*w0+x1*w1, x1*w0+x2*w1, x2*w0+x3*w1]
               t0             t1           t2           t3
```
And look at t1:
```
 [h0  x0  x1  x2  x3]
     [w0 w1]
```
This is just like sliding the kernel over one row! So this is one slice of the
real operation that the real model actually does. In the real model it does
10240 of these.

So after the loop we have done the convolution, we have:
```c++
    conv_out = ggml_silu(ctx0, conv_out);
    conv_out = ggml_reshape_3d(ctx0, ggml_cont(ctx0, conv_out), n_embd, hc, n_tokens);
    cb(conv_out, "ple_conv_out", il);

    return ggml_add(ctx0, hidden, ggml_add(ctx0, gated, conv_out));
```
And notice the last line is where `hidden`, that is `res_hc` is actually updated.

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

