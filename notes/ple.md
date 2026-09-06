### Per-Layer Embeddings (PLE)
This is a feature that enables lookup of recalled facts/information by using
a lookup table instead of processing layers in the model. So this lookup table
can live in CPU memory instead of GPU memory and be memory mapped.

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
            // for xor, but for us it is important as we want to preserve order
            // of token ids or otherwise "not good" and "good not" would xor to
            // the same value.
            uint64_t mixed = (uint64_t) ctx[0] * hp.ple_layer_multipliers[0];

            // The following loop will handle both bigram and trigram, notice
            // we are using n in this loop which will be 2 for bigrams but 3
            // for trigrams.
            for (int64_t j = 1; j < n; ++j) {
                // we xor with mixed (which is the first token id times the
                // first position multiplier.
                mixed ^= (uint64_t) ctx[j] * hp.ple_layer_multipliers[j];
            }
            // mixed is now a hash for this pair.

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

After the `build_ple` we have the following:
```console
        ggml_tensor * inject = nullptr;
        ggml_tensor * cur = build_hc_mix(res_hc,
                model.layers[il].hc_attn_norm,
                model.layers[il].hc_attn_down,
                model.layers[il].hc_attn_up,
                model.layers[il].hc_attn_inject,
                &inject, il);
```
And recall that we updated `res_hc` previously. Now what is going to happen here
is that the this layer gets to "decide via learned weights how much to pull for
each stream" which is the intuition but and we will see how this actually works
in `build_hc_mix`. Just keep in mind that the tensor `cur` that is returned
from this function is what will be passed to the linear-attention or
full-attention.
```c++
ggml_tensor * llama_model_qwen4exp::graph::build_hc_mix(
        ggml_tensor *  x,        // res_hc
        ggml_tensor *  w_norm,   // hc_attn_norm
        ggml_tensor *  w_down,   // hc_attn_down
        ggml_tensor *  w_up,     // hc_attn_up
        ggml_tensor *  w_inject, // hc_attn_inject
        ggml_tensor ** inject,   // is initially nullptr but is a reference.
        int            il) {

    const int64_t hc     = hparams.dsv4_hc_mult;
    const int64_t hc_dim = hc * n_embd;
    const int64_t nt     = x->ne[2];  // number of tokens, why not n_tokens?
```
The prompt I used was "What is the capital of Sweden?" which is 42 tokens:
```console
(gdb) p hc
$2 = 4
(gdb) p hc_dim
$3 = 10240
(gdb) p nt
$1 = 42

(gdb) p x->ne
$6 = {2560, 4, 42, 1}
```
Recall that x is `res_hc` the hyper streams and notice that we have 4 streams
per token. This is different from a normal residual connection where just have
one residual stream (per token as well) that gets updated (x = x + f(x)).
```console
Standard Transformer (per token t):
  [2560] ──────────────────────────────────────────► (1 highway)

Hyper-Connections (per token t):
  Stream 0: [2560] ───┐
  Stream 1: [2560] ───┼── (mixed by build_hc_mix) ──► (4 highways)
  Stream 2: [2560] ───┤
  Stream 3: [2560] ───┘
```

And we will first normalize the all of the hyper streams:
```c++
    // grouped RMSNorm: reduce over one stream, then scale all streams with the [hc_dim] gamma
    // the converter folded each gamma to (1 + w)
    ggml_tensor * xn = ggml_rms_norm(ctx0, x, hparams.f_norm_rms_eps);
```
Each stream is normalized against its own magnitude so they don't effect one
another. This operation only does the division step of RMSNorm so it will do
the x_i = x_i / RMS(x) part only.

In the conversion script (qwen4exp.py) we have the following:
```python
    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        ...
        # Gemma zero-centred gammas the inherited norm.weight rule misses
        if name.endswith((".ple.norm_key.weight", ".ple.norm_query.weight", ".ple.norm_conv.weight",
                          ".indexer.q_layernorm.weight", ".indexer.k_layernorm.weight")):
            return [(self.map_tensor_name(name), data_torch + 1)]
        ...
```
So this is where the + 1 happens as mentioned in the comment which is what the
forward pass in the pytorch implementation does, but here we do it upfront at
conversion time.

And now that we have normalized each stream we can reshape into
`[10240, 42]` (remember that this is `n_embd` * 4, 2560*4=10240):
```c++
    xn = ggml_reshape_2d(ctx0, xn, hc_dim, nt);
```
One thing to note is that because we are doing a reshape here before the
multiplation (below) there will be another node in the compute graph and I think
this will prevent this RMSNorm operation to be fused.
```console
(gdb) p xn->ne
$10 = {10240, 42, 1, 1}
```
So for each token we now have the four streams in one row.

Then we have gamma `hc_attn_norm/w_norm`:
```console
(gdb) p w_norm->ne
$11 = {10240, 1, 1, 1}
```
```c++
    xn = ggml_mul(ctx0, xn, w_norm);
    cb(xn, "hc_norm", il);
```


After the RMS normalization we have have the down projection:
```c++
    ggml_tensor * lo = build_lora_mm(w_down, xn);
```
```console
(gdb) p w_down->ne
$14 = {10240, 320, 1, 1}
(gdb) p xn->ne
$15 = {10240, 42, 1, 1}

             w_down
0   [0                   10239]
               .
               .
41  [0                   10239]


              xn
0   [0                   10239]
               .
               .
               .
319 [0                   10239]

          lo
 0  [0          319]
           .
           .
           ..
 41 [0          319]
```
So for each row we have performed a dot product over the 10240 dimenions so this
has summed all four hyper connection streams for each token.
```c++
ggml_tensor * llm_graph_context::build_lora_mm(
          ggml_tensor * w,             // w_down (hc_attn_down)
          ggml_tensor * cur,           // xn
          ggml_tensor * w_s) const {   // per tensor scaling
    // so first the matrix multiplication is performed.
    ggml_tensor * res = ggml_mul_mat(ctx0, w, cur);

    // this has a default value of ggml_tensor * w_s = nullptr) and this is
    // the per tensor scaling which is applied.
    if (w_s) {
        res = ggml_mul(ctx0, res, w_s);
    }

    for (const auto & lora : *loras) {
        llama_adapter_lora_weight * lw = lora.first->get_weight(w);
        if (lw == nullptr) {
            continue;
        }

        const float adapter_scale = lora.second;
        const float scale = lw->get_scale(lora.first->alpha, adapter_scale);

        ggml_tensor * ab_cur = ggml_mul_mat(
                ctx0, lw->b,
                ggml_mul_mat(ctx0, lw->a, cur)
                );

        ab_cur = ggml_scale(ctx0, ab_cur, scale);
        res = ggml_add(ctx0, res, ab_cur);
    }

    return res;
}
```
So this is a down projection from 10240 to 320 dimensions, we don't have a per
tensor scale so that is skipped and no loras:
```console
(gdb) p res->ne
$19 = {320, 42, 1, 1}
```
Next we have the non-linear operation which is SiLU in this case:
```c++
    lo = ggml_silu(ctx0, ggml_scale(ctx0, lo, 1.0f / (float) hc));
```
To understand the scaling we are doing here, lets take a smaller example to try
to understand this.
```console
n_embd = 1 (instead of 2560)
hc     = 4
w_down = [4, 1, 1, 1] (shape)

xn     = [0.9 ,  0.9,  0.9   0.9]  (the values will actually be the same for layer 0)
w_down = [0.40, 0.35, 0.50, 0.45]  (values as opposed to the shape above)

ggml_mul_mat(w_down, xn):
    
    [0.40, 0.35, 0.50, 0.45]  [0.9] = 0.40*0.9 + 0.35*0.9 + 0.50*0.9 + 0.45*0.9
                              [0.9] = 0.9 (0.40 + 0.35 + 0.50 + 0.45)
                              [0.9] = 0.9 * 1.70
                              [0.9] = 1.53
```
So that would be the dot product for this single token. Now imagin that we had
a version of this model where the hyper parameter hc was 1 instead of 4.
```console
n_embd = 1 (instead of 2560)
hc     = 1
w_down = [1, 1, 1, 1] (shape)

xn     = [0.9]
w_down = [0.40]

ggml_mul_mat(w_down, xn):
      [0.40] [0.9] = 0.40 * 0.9 = 0.36
```
Notice that we had the same values but only difference is that the number of
hyperconnections is now 1 instead of 4. But we get 0.36 instead of 1.53 which
is more that 4x difference because hc=4 summed four comparable terms.
Now, if we pass these to SiLU:
```console
silu(x) = x * sigmoid(x)

silu(1.53) = 1.53 * sigmoid(1.53) = 1.53 * 0.822 ≈ 1.258
silu(0.36) = 0.36 * sigmoid(0.36) = 0.35 * 0.589 ≈ 0.212
```
1.258 is way out in SiLU's saturation region and is almost like a pass through.
0.212 is still down near the origin where SiLU is still curving hard.
So we have the same underlying information but very different nonlinear output
which is an artifact of hc and not something the model learned.
Now, lets look what happed if we divide by hc:
```console
lo_scaled = 1.53 / 4 = 0.3825
silu(0.3825) = 0.3825 * sigmoid(0.3825) = 0.3825 * 0.594 ≈ 0.227
```
Notice that this is much closer. So just keep this in mind we are scaling the
values before we call `ggml_silu` to do just what we showed above, we are doing
an elemenent wise scaling, dividing by hc, and the values are the dot product,
and then passing that scaled tensor to silu:
```c++
    lo = ggml_silu(ctx0, ggml_scale(ctx0, lo, 1.0f / (float) hc));
```
To recap, we have taking the 4 hyper connection streams, down projected to them
into a smaller dimension, which we as otherwise it would require a lot of memory
and compute. We then scale those values like we just discussed and then pass
them through SiLU. And SiLU will filter like this:
```console
                           z
silu(z) = z * σ(z) =    --------
                         1 + e^-z
```
So if we passed in 0.3825 we would get:
```console
                                        0.3825
silu(0.3825) = 0.3825 * σ(0.3825) =  ------------- = 0.22738
                                      1 + e^-0.3825
```
And lets say we have a negative value:
```console

                               -10
silu(-10) = -10 * σ(-10) =  ----------------- = -0.0004539
                               1 + e^-(-10)

```
So in our case this will act as a filter on all the 320 values features for each
token to determine which to suppress and which to keep. So if the value is positive
then it is kept, and if it is negative it is silenced and set to zero.

Next, after the filtering we have an up projection back to the 10240 dimension,
and a sigmoid activation operation:
```c++
    ggml_tensor * gate = ggml_sigmoid(ctx0, build_lora_mm(w_up, lo));
```
```console
(gdb) p w_up->ne
$31 = {320, 10240, 1, 1}
(gdb) p lo->ne
$32 = {320, 42, 1, 1}

(gdb) p build_lora_mm(w_up, lo, 0x0)->ne
$33 = {10240, 42, 1, 1}
```
So this brings us back to our "normal" shape with a dimension of 10240. Now, instead
if SiLU we will just use sigmoid to bound each channel to a value between 0.0
and 1.0. This will the act like a gate for each channel in a row of 10240. So
for a single token, which has 4 hyperconnection streams, this is essentially
allowing/rejecting information from all 4 hyperconnections streams when the gate
is later used on the current input (xn).

The gate is then used on the current input which is what uses the gate:
```c++
    ggml_tensor * gated = ggml_mul(ctx0, xn, gate);
```
```console
(gdb) p gated->ne
$34 = {10240, 42, 1, 1}
```
Then we reshaped this into a 3d tensor, spliting out the hyper connections:
```c++
    gated = ggml_reshape_3d(ctx0, gated, n_embd, hc, nt);
```
```console
(gdb) p gated->ne
$35 = {2560, 4, 42, 1}
```
If we think of this as just an array in memory:
```console
[0 ... 2559][2560 ... 5119][5120 ... 7679][7680 ... 10239][ ... ][ ... ] ...
   t0 s0        t0 s1           t0 s2          t0 t3       t1 s0  t1 s1  ...
|------------------------------------------------------->|
              stride 40960
```
We then create a 2d view into the gated tensor of shape [2560, 42] with a stride 
of stride of 40960 (10240 * 4 = 40960 bytes):
```c++
    // collapse the streams by their mean
    ggml_tensor * mixed = ggml_view_2d(ctx0, gated, n_embd, nt,
            ggml_row_size(gated->type, n_embd) * hc, 0);
    mixed = ggml_cont(ctx0, mixed);
```
And we are making this first view contiguous so that the optimized path will
be taken below.
Next we have a loop from 1 up to hc (4) as mixed is already the first row:
```c++
    for (int64_t c = 1; c < hc; ++c) {
        ggml_tensor * s = ggml_view_2d(ctx0, gated, n_embd, nt,
                ggml_row_size(gated->type, n_embd) * hc,
                ggml_row_size(gated->type, n_embd) * c);
        mixed = ggml_add(ctx0, mixed, s);
    }
```
The tensor returend by `ggml_add` is contiguous automatically so we don't need
another `ggml_cont`, we only need that for the first slice. And mixed is what
this function is going to return, it will be the input, the mixed input with
the hyper connection streams and what will be passed to the linear-attention
or full-attention.
After that we scale once more:
```c++
    mixed = ggml_scale(ctx0, mixed, 1.0f / (float) hc);
```
And again this is to take into consideration a model what has a different hc,
just like we discussed above.

The last thing before mixed is returned is:
```c++
    if (inject) {
        // w_inject is hc_attn_inject
        *inject = build_lora_mm(w_inject, xn);
        cb(*inject, "hc_inject", il);
    }
```
gT
```console
(gdb) p w_inject->ne
$48 = {10240, 4, 1, 1}

(gdb) p xn->ne
$49 = {10240, 42, 1, 1}

(gdb) p (*inject)->ne
$53 = {4, 42, 1, 1}

0  [0 ... 3]
       .
       .
       .
41 [0 ... 3]
```
So for each hyper connection, which remember there are 4 streams, this produces
a single scalar value for each stream. It value represents how relevant this
block's output to each stream. So this is not related to the input for the
rest of the current layer but if for the output to the next layer and determines
how much of each stream should be passed along.
Lets say we have the following values:
```
token 0 [0.23, -3.3, 0.93, 8.0]

Stream 0: (w≈1.03) essentially like a normal residual connection (1.0 is neuaral)
Stream 1: (w≈0.61) suppressed to about 61% of normal, so this will dampen how
          much this block's (layer's) output reaches stream 1. So stream1 might
          already be h olding a fact/result that should pass through mostly unchanged.
Stream 2: (w≈1.12) mildly amplified.
Stream 3: (w≈1.76) strongly amplified. This value is close to the ceiling of 2.0
          This layers output is especially relevant to stream 3.
```
I'll return to this inject tensor later when it get used.

And after that mixed is returned and we are finished with this function:
```c++
    return mixed;
}
```
So back in `llama_model_qwen4exp::graph::graph` we have:
```c++
        ggml_build_forward_expand(gf, cur);

        if (hparams.is_recr(il)) {
            cur = build_layer_attn_linear(inp->get_recr(), cur, il);
        } else {
            cur = build_layer_attn(inp->get_attn(), mctx_hyb, cur, inp_pos, sections, il);
        }
```

_wip_

