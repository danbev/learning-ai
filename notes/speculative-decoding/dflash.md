## D-Flash speculative decoding

llama.cpp pull request: https://github.com/ggml-org/llama.cpp/pull/22105

What block diffusion flash (D-Flash) addresses is the fact that when we have a
draft model it is still an autoregressive model, like Medusa and Eagle, which
processes one token at a time which means that there is a limit to how good
these types of models can perform. The D in D-Flash stands for diffusion.

So diffusion will generate all the "stuff" at once if for example we think of
an image. But in an LLM we don't have a fixed size output to generate which is
a requirement of a diffusion model (I think). So what D-Flash does is it creates
blocks of tokens of a fixed size and then it generates those blocks in one
forward process or the diffusion model.

So the flow is something like this, we first process the initial prompt using
the target model, just like any normal inference. But the model will extract a
number of layer's outputs as features, which it will then be merged and
normalized to fit the dimension of the draft model. These features will then be
used with cross attention when drafting, stering the draft model towards good
outputs. So the draft model is influenced, given context, by the target model
and this should produce better draft predictions.

Then the draft model will predict a block of tokens. So lets say we are going
to predict 4 tokens:
```
slot :   0        1      2      3
token: [anchor] [MASK] [MASK] [MASK]
```
This is like the noise in the diffusion process. The draft model will use the
cross attention features to predict something better than random tokens.

Now, these are predicted in a single forward pass but there are usually a few
steps (are these layers of actual full diffusion processes?).
The draft model looks at all the slots simultaneously using bidirectional
attention.

The draft model will predict a probablity distribution for every mask at the
same time.
```
slot 0: "diffusion" (80% confidence)
slot 1: "process" (40% confidence)
slot 2: "is" (75% confidence)
slot 3: "cool" (30% confidence)
```
Now, the draft model might choose the top two predictions that it is most
certain about, so in this case "diffusion" and "is". And the others will be kept
as MASK tokens.

The model then runs another quick pass but now it has more information/context,
it has the predictions from the first pass, in addition to the cross attention
features from the target model.
```
slot 1: "process" (now 95% confidence)
slot 3: "clear" (now 90% confidence)
```
After this the masks have been replaced with actual tokens, so we have generated
4 tokens in 2 passes:
```
Final Draft Output: diffusion process is clear
```

Alright lets try to understand this better. We will have a block of tokens, lets
say 8. This is a number of latent vectors.
```console
 0 [0   ... h_dim]
 1 [0   ... h_dim]
 2 [0   ... h_dim]
 3 [0   ... h_dim]
 4 [0   ... h_dim]
 5 [0   ... h_dim]
 6 [0   ... h_dim]
 7 [0   ... h_dim]
```
So initially this would just be noise, like in stable diffusion where we start
with guassian noise. But here the positions are initialized using masked embeddings.
Each slot will be filled with a masked token embedding:
```
 0 [0   ... h_dim]  <masked token or blank vector>
 1 [0   ... h_dim]  <masked token or blank vector>
 2 [0   ... h_dim]  <masked token or blank vector>
 3 [0   ... h_dim]  <masked token or blank vector>
 4 [0   ... h_dim]  <masked token or blank vector>
 5 [0   ... h_dim]  <masked token or blank vector>   
 6 [0   ... h_dim]  <masked token or blank vector>
 7 [0   ... h_dim]  <masked token or blank vector>
```

### Nemotron 3.5 Lightning DFlash walkthrough
If we look in this models config.json we find:
```console
  "target_layer_ids": [
    1,
    5,
    19,
    29,
    41,
    51
  ],
```
This identifies the output of layer 1, 5, 19, etc.

When the models hyperparameters are loaded these target layer ids are extracted:
```c++
    if (!ml.get_arr(LLM_KV_TARGET_LAYERS, target_layer_ids, false)) {
        throw std::runtime_error("DFlash model requires 'target_layers' in GGUF metadata");
    }
```
If I print these in llama.cpp I get:
```console
(gdb) p target_layer_ids
$16 = std::vector of length 6, capacity 6 = {2, 6, 20, 30, 42, 52}
```
Where we use 2, to mean the input to layer 2, which is the output of layer 1 so
we are referring to the same layers even thought they are different in the original
models configuration. The adding of +1 is done in conversion/qwen.py:
```python
        target_layer_ids = dflash_config.get("target_layer_ids", [])
        if target_layer_ids:
            extract_layer_ids = [i + 1 for i in target_layer_ids]
            self.gguf_writer.add_target_layers(extract_layer_ids)
```
```c++
    hparams.n_embd_inp_enc_impl = (uint32_t) target_layer_ids.size() * hparams.n_embd;
```
```console
(gdb) p hparams.n_embd_inp_enc_impl
$19 = 16128
```
```console
number of selected target layers = 6
target hidden size               = 2688

  encoder input size = 6 x 2688
                     = 16128
```
That matches the fc.weight tensor:
```
  fc.weight: [16128, 2688]
```

The following is to give a general overview of the process that we will later
go through in details. But I think it helps to have a big picture view of what
is doing on. We are using llama-server for this debugging session.

So recall that in dflash the above 6 layers are saved when the target model runs
it decoding. As we will see later, when we step through the code, when the dflash
speculative decoder is initalized it will have access to both the draft and the
target model. This enables it to set the target layer ids on the target model's
llama_context, enabling them to be availble when the target models computation
graph is build all inputs to the layers which are stored (the ggml_tensor*), and
after the graph is built, set_outputs is called and these layers will set as
output tensors. This means that they will be part of the host output buffer and
copied from the device to the host after process_ubatch has been called.

So llama-server will first call `llama_decode` on the target model which will store
the above layers as outputs. llama-server will the call the speculative
decoders, which for dflash means that it can access these layers from the
target model using `llama_get_embeddings_layer_inp` and then copying them into
a batch as embeddings which it will use to encode them.

At runtime, llama.cpp gathers one 2688-value hidden state from each selected target layer:
```console
  layer 2 input:   2688 values
  layer 6 input:   2688 values
  layer 20 input:  2688 values
  layer 30 input:  2688 values
  layer 42 input:  2688 values
  layer 52 input:  2688 values
                   ----
  concatenated:   16128 values
```
fc.weight then projects that combined vector down to the DFlash hidden size:
```console
  16128 -> 2688
```
So this is the output of these layers from the target model, which are then
concatenated and down projected. The output of the projection is a 2688 element
representation of the token that the target model processed (at different layers)
so not just the final score.

This fusion is performed for every target token being processed, not only the
latest anchor token. Therefore, the draft KV cache contains a history of fused
target representations:
```console
  token 0 fused representation
  token 1 fused representation
  ...
  last committed token fused representation
```
Then the diffusion block begins with the last committed token ID and uses
attention over that injected history to predict the upcoming masked positions.

```
  six target hidden states
          |
          v
  concatenate and apply fc
          |
          v
  fused 2688-value vector
          |
          v
  DFlash K/V projections
          |
          v
  store in DFlash KV cache
```
When DFlash later drafts a block, its token input looks roughly like:
```console
  [last real token | MASK | MASK | ...]
```
The first entry is the anchor token. It is initially represented using the token
embedding for the last real token, not the fused fc vector.

Next we have the following:
```c++
 24     ml.get_key(LLM_KV_HYPER_CONNECTION_COUNT, hparams.dsv4_hc_mult, false);
```
And this is a parameter that gives the number of manifold-constrainded
hyper-connections (mHC). This is not used in this model but I need to look into
this as some point. TODO: look into mHC.

Next lets look at the loading of tensors:
```c++
void llama_model_dflash::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const int64_t n_embd_inp = hparams.n_embd_inp_enc();
```
```console
(gdb) p n_embd_inp
$3 = 16128
(gdb) p n_embd
$4 = 2688
```
And these are familiar from before and will be used to load the fully connected
(fc) tensors:
```c++
    fc   = create_tensor(tn(LLM_TENSOR_FC, "weight"), { n_embd_inp, n_embd }, 0);
    fc_s = create_tensor(tn(LLM_TENSOR_FC, "scale"),  { 1 }, TENSOR_NOT_REQUIRED);
```
And recall that 0 mean:
```console
  required             = yes
  allow reshape        = no
  duplicated tensor    = no
  other special flags  = no
```
Next we have:
```c++
    const struct ggml_tensor * markov_meta = ml->get_tensor_meta("markov_w1.weight");
    if (markov_meta) {
```
The `markov_w1.weight` tensor would only be in a dspark model as dflash does not
have that type of sequential small network (nor does dflash have a confindance
projection tensors which are created in body of the if statement:
```c++
        const int64_t dspark_markov_rank = markov_meta->ne[0];

        dspark_markov_w1 = create_tensor(tn(LLM_TENSOR_DSPARK_MARKOV_W1, "weight"),
            { dspark_markov_rank, n_vocab }, 0);
        dspark_markov_w2 = create_tensor(tn(LLM_TENSOR_DSPARK_MARKOV_W2, "weight"),
            { dspark_markov_rank, n_vocab }, 0);

        dspark_conf_proj   = create_tensor(tn(LLM_TENSOR_DSPARK_CONF_PROJ, "weight"),
            { n_embd + dspark_markov_rank, 1 }, 0);
        dspark_conf_proj_b = create_tensor(tn(LLM_TENSOR_DSPARK_CONF_PROJ, "bias"),
            { 1 }, TENSOR_NOT_REQUIRED);
    }
```
The check is need as both dflash and dspark use the same general.architecture:
```console
  general.architecture = dflash
```
Which makes sense as dspark is an extension of dflash. So this will be skipped
for this session and we will proceed with:
```c++
    fc   = create_tensor(tn(LLM_TENSOR_FC, "weight"), { n_embd_inp, n_embd }, 0);
    fc_s = create_tensor(tn(LLM_TENSOR_FC, "scale"),  { 1 }, TENSOR_NOT_REQUIRED);
```
And we saw these before and that they are related to the merging/forging of the
target model layer specified by the target layer ids we looked at before. And
the scale is the global tensor scale (the block scale was merged into the fc.weight
in the convertion of the model).

Next we have the creation of the mHC tensors but I'm skipping the for now as
they are not used in this model.

After that we have:
```c++
    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);

        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), { n_embd, n_embd_head_k * n_head }, 0);
        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), { n_embd, n_embd_k_gqa }, 0);
        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), { n_embd, n_embd_v_gqa }, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd_head_k * n_head, n_embd }, 0);

        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), { n_embd_head_k }, 0);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), { n_embd_head_k }, 0);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), { n_embd }, 0);
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), { n_embd, n_ff }, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff, n_embd }, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), { n_embd, n_ff }, 0);
    }
```
And we have 6 layers:
```console
(gdb) p n_layer
$10 = 6
```
This is the drafting models "main" layers if I'm not mistaken. So this looks 6
layers of self attention.
 ```console
self-attention
        +
feed-forward network
```
tensors for each layer:
```console
layer.attn_norm
layer.wq
layer.wk
layer.wv
layer.wo
layer.attn_q_norm
layer.attn_k_norm

layer.ffn_norm
layer.ffn_gate
layer.ffn_up
layer.ffn_down
```
```console
  input hidden state
         |
         +-------------------------------+
         |                               |
         v                               |
  attention RMSNorm                      |
         |                               |
         v                               |
  Q, K and V projections                 |
         |                               |
         v                               |
  self-attention                         |
         |                               |
         v                               |
  attention output projection            |
         |                               |
         +------------ add residual -----+
         |
         +-------------------------------+
         |                               |
         v                               |
  FFN RMSNorm                            |
         |                               |
         v                               |
  gated feed-forward network             |
         |                               |
         +------------ add residual -----+
         |
         v
  output hidden state
```

### attn_norm.weight
Normalizes the layer input before self-attention:
```console
  normalized_input = RMSNorm(input)
```

### attn_q.weight
Projects the normalized input into query vectors:
```console
  Q = Wq x normalized_input
```
The query asks:
```console
Which stored token positions are relevant to the current position?
```

### attn_k.weight
Projects the input into key vectors:
```console
  K = Wk x normalized_input
```
Keys describe what each position can be matched against.

### attn_v.weight
Projects the input into value vectors:
```console
  V = Wv x normalized_input
```

Values contain the information returned when a key is selected.

### attn_q_norm.weight and attn_k_norm.weight
These apply RMS normalization to each Q and K head before attention:
```console
  Q = RMSNorm(Q)
  K = RMSNorm(K)
```
This is often called QK normalization. It helps keep attention scores stable.

### attn_output.weight
After attention combines the value vectors, this projects the result back into
the model hidden size:
```console
  attention_output = Wo x attended_values
```

The result is then added to the layer input through the residual connection.

## Feed-forward tensors
After attention, the layer has a gated MLP.

### ffn_norm.weight
Normalizes the state before the feed-forward network.

### ffn_gate.weight
Produces the gate branch:
```console
  gate = Wgate x normalized_input
```

### ffn_up.weight
Produces the value branch:

### Activation and gating
The two branches are combined approximately as:
```console
  hidden = SiLU(gate) * up
```

### ffn_down.weight
Projects the expanded FFN representation back to the model width:
```console
  ffn_output = Wdown x hidden
```
The FFN output is then added to its input through another residual connection.

## Dimensions in this model
The draft hidden size is:
```console
  n_embd = 2688

  The FFN size is:

  n_ff = 6144

  So the FFN expands and contracts:

  2688 -> 6144 -> 2688

  For attention:

  32 query heads
  2 key/value heads
  128 values per head
```
This is grouped-query attention:
32 query heads share 2 key/value heads

Each KV head is shared by 16 query heads.


After all tensors have been loaded, `build_arch_graph` will later be called to
build the computation graphs:
```console
#0  llama_model_dflash::build_arch_graph (this=0xaaaab6f229f0, params=...)
    at /home/danbev/work/llama.cpp/src/models/dflash.cpp:176
#1  0x0000fffff1c17e0c in llama_model::build_graph (this=0xaaaab6f229f0, params=...)
    at /home/danbev/work/llama.cpp/src/llama-model.cpp:244
```

```c++
std::unique_ptr<llm_graph_context> llama_model_dflash::build_arch_graph(const llm_graph_params & params) const {
    switch (params.gtype) {
        case LLM_GRAPH_TYPE_ENCODER:
            return std::make_unique<graph<true>>(*this, params);
        case LLM_GRAPH_TYPE_DEFAULT:
        case LLM_GRAPH_TYPE_DECODER:
            if (hparams.dsv4_hc_mult > 0) {
                return std::make_unique<graph_dsv4>(*this, params);
            }
            return std::make_unique<graph<false>>(*this, params);
        default:
            GGML_ABORT("invalid graph type");
    };
}
```
In our case the graph type is `LLM_GRAPH_TYPE_DEFAULT` so and our model does not
support mHC so we will just call:
```c++
            return std::make_unique<graph<false>>(*this, params);
```

```console
(gdb) ptype graph<false>
type = struct llama_model_dflash::graph<false> : public llm_graph_context {

    graph(const llama_model &, const llm_graph_params &);
    ggml_tensor * build_inp_embd_enc(void) const;
}
```
And this will call:
```c++
template <>
llama_model_dflash::graph<false>::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {

```
So this is an explicit specialization of the graph class template constructor
for false.
```c++
    ggml_tensor * inp_pos  = build_inp_pos();
```
This graph can accept a micro batch (ubatch) with an embedding, in addition to
tokens:
```
    if (ubatch.embd) {
        auto inp = std::make_unique<llm_graph_input_embd>(n_embd);
```
inp->embd contains the selected target-layer states after concatenation, fc
projection, and encoder normalization. This decoder branch turns those fused
vectors into K/V cache entries for every DFlash layer.

The full path is:
```console
  selected target hidden states
          |
          v
  concatenate
          |
          v
  fc projection
          |
          v
  encoder RMSNorm
          |
          v
  inp->embd in this decoder graph
```
The this is then later set as input in speculative.cpp:
```c++
                for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
                    const float * layer = llama_get_embeddings_layer_inp(ctx_tgt, (uint32_t) target_layer_ids[k]);
                    ...

                // fuse extracted features through DFlash encoder
                llama_batch enc_batch = {
                    /*.n_tokens =*/ n_chunk,
                    /*.token    =*/ nullptr,
                    /*.embd     =*/ features_buf.data(),
                    /*.pos      =*/ nullptr,
                    /*.n_seq_id =*/ nullptr,
                    /*.seq_id   =*/ nullptr,
                    /*.logits   =*/ nullptr,
                };
```
```console
(gdb)  p *ubatch.token@ubatch.n_tokens
$23 = {0, 0, 0, 0}
```

So understand the overall flow, lets debug llama-server. So we will first start
the server and set a breakpoint in the servers code, just before it decodes the
prompt:
```console
(gdb) bt
#0  server_context_impl::decode (this=0xaaaaabe94bf0, n_batch=@0xffffffff58b8: 2048, off=0, batch_view=...)
    at /home/danbev/work/llama.cpp/tools/server/server-context.cpp:3560
#1  0x0000fffff763d2d0 in server_context_impl::update_slots (this=0xaaaaabe94bf0)
    at /home/danbev/work/llama.cpp/tools/server/server-context.cpp:2772
#2  0x0000fffff763420c in server_context_impl::init()::{lambda()#1}::operator()() const (__closure=0xaaaaabe94d38)
    at /home/danbev/work/llama.cpp/tools/server/server-context.cpp:1361
#3  0x0000fffff766803c in std::__invoke_impl<void, server_context_impl::init()::{lambda()#1}&>(std::__invoke_other, server_context_impl::init()::{lambda()#1}&) (__f=...) at /usr/include/c++/13/bits/invoke.h:61
#4  0x0000fffff7660b40 in std::__invoke_r<void, server_context_impl::init()::{lambda()#1}&>(server_context_impl::init()::{lambda()#1}&) (__fn=...) at /usr/include/c++/13/bits/invoke.h:111
#5  0x0000fffff7656e0c in std::_Function_handler<void (), server_context_impl::init()::{lambda()#1}>::_M_invoke(std::_Any_data const&) (__functor=...) at /usr/include/c++/13/bits/std_function.h:290
#6  0x0000fffff74df114 in std::function<void ()>::operator()() const (this=0xaaaaabe94d38)
    at /usr/include/c++/13/bits/std_function.h:591
#7  0x0000fffff75a8a1c in server_queue::start_loop (this=0xaaaaabe94c08, idle_sleep_ms=-1000)
    at /home/danbev/work/llama.cpp/tools/server/server-queue.cpp:163
#8  0x0000fffff760bb20 in server_context::start_loop (this=0xffffffff7098)
    at /home/danbev/work/llama.cpp/tools/server/server-context.cpp:4025
#9  0x0000fffff74cc4b0 in llama_server (params=..., argc=16, argv=0xffffffffece8)
    at /home/danbev/work/llama.cpp/tools/server/server.cpp:520
#10 0x0000fffff74c89f4 in llama_server (argc=16, argv=0xffffffffece8)
    at /home/danbev/work/llama.cpp/tools/server/server.cpp:112
#11 0x0000aaaaaaaa17b4 in main (argc=16, argv=0xffffffffece8) at /home/danbev/work/llama.cpp/tools/server/main.cpp:4


(gdb) p batch_view.token[0]@batch_view.n_tokens
$10 = {10, 25708, 1010, 11, 1010}

(gdb) p batch_view.pos[0]@batch_view.n_tokens
$11 = {0, 1, 2, 3, 4}
```
Now, actually at this stage the speculative decoding will have been initialized
already. When server-context.cpp has its `load_model` function called we have the
init of speculative samplers:
```c++
    bool load_model(common_params & params) {
        ...

        // try speculative decoding
        if (ctx_tgt_seq_rm_type != COMMON_CONTEXT_SEQ_RM_TYPE_NO) {
            try {
                spec.reset(common_speculative_init(params_base.speculative, params_base.n_parallel));
            } catch (const std::exception & e) {
                SRV_ERR("failed to initialize speculative decoding context: %s\n", e.what());
            }
        }
```
For dflash this will call

```c++
common_speculative * common_speculative_init(common_params_speculative & params, uint32_t n_seq) {
    ...

            case COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_dflash>(config.params, n_seq));
                break;
            }
```
And the constructor looks like this:
```c++
    common_speculative_impl_draft_dflash(const common_params_speculative & params, uint32_t n_seq,
            common_speculative_type type = COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH)
        : common_speculative_impl(type, n_seq)
        , params(params.draft)
        , is_dspark(type == COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK) {
        auto * ctx_tgt = this->params.ctx_tgt;  // target llama context
        auto * ctx_dft = this->params.ctx_dft;  // draft llama context

        const llama_model * model_dft = llama_get_model(ctx_dft);
        const llama_model * model_tgt = llama_get_model(ctx_tgt);

        target_layer_ids   = llama_model_target_layer_ids  (model_dft);
        target_layer_ids_n = llama_model_target_layer_ids_n(model_dft);
```
We can see here that we are getting a pointer target layer ids from the draft
model.

Next we get the embeddings dimension from the target model, and from the draft
model, and the embedding of the encoder will be the target embedding dimensions
times the number of target layers (features) that are extracted:
```c++
        n_embd_tgt    = llama_model_n_embd(model_tgt);
        n_embd_dec    = llama_model_n_embd(model_dft);
        n_embd_enc    = (int32_t) target_layer_ids_n * n_embd_tgt;
```
Then the block size is retrieved from the draft models metadata, dflash.block_size.
The block is 16 in this case and it means this number of tokens are processed
together by one diffusion block:
```console
[anchor token, mask, mask, ... , mask]
  0             1      2           15
```
Where anchor is the most recent token from the target model. So the max number
of tokens it can propose is 15 tokens (16-1, excluding the achor token).
Interesting is that dspark can propose 16 tokens, I need to look into that.

Further down we then have:
```c++
        batch        = llama_batch_init(llama_n_batch(ctx_dft), 0,          n_seq);
        batch_inject = llama_batch_init(llama_n_batch(ctx_dft), n_embd_dec, n_seq);
```
The `batch` is used for drafting, and it does not include any embeddings (the
second argument), 0 that is. 

But `batch_inject` does specify that it will be using `n_embd_dec` (2688)
embeddings and this will later be used to pass the projected token features into
the dflash decoder to populate the KV cache.

All sequences are given a sampler with a top k (10) sampler.
```c++
        smpls.resize(n_seq);
        for (auto & s : smpls) {
            common_params_sampling sparams;
            sparams.no_perf  = false;
            sparams.top_k    = 10;
            sparams.samplers = { COMMON_SAMPLER_TYPE_TOP_K };
            s.reset(common_sampler_init(model_dft, sparams));
        }
```
Why 10? This is also the same value that is used for backend samplers if enabled.

Next we iterate over all the target layer ids, and use the target models llama
context (remember, the target model need to know which layers to extract and
this information, the metadata is part of the draft model):
```c++
        // turn on extraction of the target layers' input embeddings
        for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
            llama_set_embeddings_layer_inp(ctx_tgt, (uint32_t) target_layer_ids[k], true);
        }
```
```c++
void llama_context::set_embeddings_layer_inp(uint32_t lid, bool enable) {
    LLAMA_LOG_DEBUG("%s: lid = %d, enable = %d\n", __func__, lid, enable);

    GGML_ASSERT(lid <= model.hparams.n_layer());

    cparams.embeddings_layer_inp[lid] = enable;

    // note: without this reserve, the draft acceptance drops to zero. not sure why - this is unexpected
    sched_need_reserve = true;
}
```
So `embedding_layers_inp` is a vector of boolean values, one for each layer in
the model. And we are setting the target layers to true. So this is how this
information is transferred from the draft model to the target model. This tells
the target context that when these target layers run, preserve their input hidden
states so dflash can get them afterward.

So if we look in nemotron-h.cpp we can see this in action:
```c++
llama_model_nemotron_h::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_build_mamba_base(params) {

    ...
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);
    ggml_build_forward_expand(gf, inpL);

    for (int il = 0; il < n_layer; ++il) {
        res->t_layer_inp[il] = inpL;
```
Notice that this code is storing the input to the layers in `t_layer_inp` and
this is done for all layers. This is only storing pointers to the tensors in
that vector. These tensors have not been set as output, so they won't all be
outputs of the graph (not yet at least).

To understand this better `graph_reserve` will call `build_graph`:
```c++
ggml_cgraph * llama_model::build_graph(const llm_graph_params & params) const {
    std::unique_ptr<llm_graph_context> llm = build_arch_graph(params);

    // add on pooling layer
    llm->build_pooling(cls, cls_b, cls_out, cls_out_b, cls_norm);

    // add backend sampling layers (if any)
    llm->build_sampling();

    // if the gguf model was converted with --sentence-transformers-dense-modules
    // there will be two additional dense projection layers
    // dense linear projections are applied after pooling
    // TODO: move reranking logic here and generalize
    llm->build_dense_out(dense_2_out_layers, dense_2_out_layers_b, dense_3_out_layers);

    llm->res->set_outputs(params);

    return llm->res->get_gf();
}
```
The graph construction is what is called by the first line, and then notice that
`llm->res->set_outputs` is called.
```c++
void llm_graph_result::set_outputs(const llm_graph_params & params) {
    ...

    if (t_h_nextn != nullptr) {
        ggml_set_output(t_h_nextn);
    }

    {
        const auto & embeddings_layer_inp = params.cparams.embeddings_layer_inp;
        for (size_t il = 0; il < embeddings_layer_inp.size(); ++il) {
            if (embeddings_layer_inp[il]) {
                GGML_ASSERT(t_layer_inp[il] != nullptr && "layer input tensor is null");
                ggml_set_output(t_layer_inp[il]);
            }
        }
    }

    ...
}
```
And this will iterate over all the `embeddings_layer_inp` as set only the target
feature layers that were set above.
```console
(gdb) p embeddings_layer_inp
$11 = std::vector<bool> of length 53, capacity 64 = {false, false, true, false, false, false, true, false, false,
  false, false, false, false, false, false, false, false, false, false, false, true, false, false, false, false,
  false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false,
  false, true, false, false, false, false, false, false, false, false, false, true}
```
After the graph execution (`process_ubatch`) `llama_decode` will call:
```c++
        extract_layer_inputs(res, n_tokens_prev, ubatch.n_tokens);
```
```c++
void llama_context::extract_layer_inputs(const llm_graph_result * res, size_t token_offset, size_t n_tokens) {
    for (uint32_t il = 0; il < cparams.embeddings_layer_inp.size(); ++il) {
        if (!cparams.embeddings_layer_inp[il]) { // skip layer if not set
            continue;
        }

        if (!embd_layer_inp[il].has_data()) {
            GGML_ABORT("output layer input buffer not allocated");
        }
        ggml_tensor * t = res->get_layer_inp((int) il);
        if (!t) {
            GGML_ABORT("layer input tensor not found");
        }

        const size_t nbytes = ggml_nbytes(t);
        const size_t nfloats = nbytes / sizeof(float);
        GGML_ASSERT(n_tokens > 0);
        GGML_ASSERT(nfloats % n_tokens == 0);

        const size_t row_floats = nfloats / n_tokens;
        const size_t dst_offset = token_offset * row_floats;
        GGML_ASSERT(dst_offset + nfloats <= embd_layer_inp[il].size);

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched.get(), t);
        GGML_ASSERT(backend != nullptr);
        ggml_backend_tensor_get_async(backend, t, embd_layer_inp[il].data + dst_offset, 0, nbytes);
    }
}
```
And that is very similar to what we did with backend sampling data as well, infact
there is just one large host buffer for all outputs.
```console
(gdb) p il
$16 = 2

(gdb) p t->ne
$15 = {2688, 5, 1, 1}
```
So this is getting the output from layer 1 (the input to layer 2), and this is
a [2688, 5] tensor (so 5 rows and 2688 columns) and that matches the number of
tokens we have in the ubatch:
```console
(gdb) p n_tokens
$21 = 5
```
And we then copy these asynchronously from the backend into `embd_layer_inp`
which is of type:
```console
(gdb) ptype embd_layer_inp
type = std::vector<buffer_view<float>>
```
And this is a view into the `buf_output` which is a host side buffer which is
used to store all the outputs (this is allocated in `reserve_outputs`).

So this is how the target feature layers are extracted and copied over to the
host, very similar to how backend sampling also works.

When dflash spec.decoder's process function is called it will be able to access
this data, and just to be clear this is just the raw hidden vector states from
those layers, they have not been processed in any way (yet):
```c++
    bool process(const llama_batch & batch_in) override {
        ...
        std::vector<int32_t> i_batch_beg(n_seq, -1);
        std::vector<int32_t> i_batch_end(n_seq, -1);

        for (int32_t k = 0; k < n_tokens; ++k) {
            GGML_ASSERT(batch_in.n_seq_id[k] == 1);

            const llama_seq_id seq_id = batch_in.seq_id[k][0];
            if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
                continue;
            }

            // unconditionally set the end and it will end up storing the
            // last index.
            i_batch_end[seq_id] = k;

            if (i_batch_beg[seq_id] < 0) {
                i_batch_beg[seq_id] = k;
            }
        }
```
So we have a batch which is just tokens and their sequence ids, this loop is
setting up so that we can find where in the list of tokens a sequence starts
and ends. This will be used below:
```c++
        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (i_batch_beg[seq_id] < 0) {
                continue;
            }

            // calculate the number of slots (as in slots in the token array in
            // the ubatch. 
            const int32_t n_rows = i_batch_end[seq_id] - i_batch_beg[seq_id] + 1;

            // this will iterate over all the n_rows, in steps of n_ubatch
            for (int32_t offset = 0; offset < n_rows; offset += n_ubatch) {
                // if the n_ubatch size was set to a value smaller than the number
                // of slots that this sequence has in the tokens array, then we
                // clamp it to the minimum.
                const int32_t n_chunk = std::min(n_ubatch, n_rows - offset);

                // gather this chunk's target features, interleaved by extract layer
                features_buf.resize((size_t) n_chunk * n_embd_enc);

                // for all of the extracted layers...
                for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
                    // get the target feature vector for this target layer
                    const float * layer = llama_get_embeddings_layer_inp(ctx_tgt, (uint32_t) target_layer_ids[k]);
                    if (!layer) {
                        GGML_ABORT("DFlash: target layer %d input not extracted.", target_layer_ids[k]);
                    }

                    for (int32_t i = 0; i < n_chunk; ++i) {
                        float       * dst = features_buf.data() + (size_t) i * n_embd_enc + k * (size_t) n_embd_tgt;
                        const float * src = layer + (size_t) (i_batch_beg[seq_id] + offset + i) * n_embd_tgt;
                        std::memcpy(dst, src, (size_t) n_embd_tgt * sizeof(float));
                    }
                }
```
This function will iterate over all the sequences that we have, just 4 in our
case as the server has 4 parallel slots.

This `features_buf` will then be passed as the embeddings encoding batch:
```c++
                // fuse extracted features through DFlash encoder
                llama_batch enc_batch = {
                    /*.n_tokens =*/ n_chunk,
                    /*.token    =*/ nullptr,
                    /*.embd     =*/ features_buf.data(),
                    /*.pos      =*/ nullptr,
                    /*.n_seq_id =*/ nullptr,
                    /*.seq_id   =*/ nullptr,
                    /*.logits   =*/ nullptr,
                };

                int32_t rc = llama_encode(ctx_dft, enc_batch);
```
```console
layer 2  [2688, 5]   [layer 2: 2688][layer 6: 2688][layer 20: 2688]...[layer 52: 2688]
layer 6  [2688, 5]
layer 20 [2688, 5]      => [16128, 5]
layer 30 [2688, 5]
layer 42 [2688, 5]
layer 52 [2688, 5]
```
```console
(gdb) p features_buf.size()
$39 = 80640
(gdb) p 16128 * 5
$40 = 80640
```
So to follow the encode call we have to remember that this is the draft model
that is doing to do this, so we should look in dflash.cpp, and the encoder is
chosen by this code:
```c++
std::unique_ptr<llm_graph_context> llama_model_dflash::build_arch_graph(const llm_graph_params & params) const {
    switch (params.gtype) {
        case LLM_GRAPH_TYPE_ENCODER:
            return std::make_unique<graph<true>>(*this, params);
        case LLM_GRAPH_TYPE_DEFAULT:
        case LLM_GRAPH_TYPE_DECODER:
            if (hparams.dsv4_hc_mult > 0) {
                return std::make_unique<graph_dsv4>(*this, params);
            }
            return std::make_unique<graph<false>>(*this, params);
        default:
            GGML_ABORT("invalid graph type");
    };
}
```
```c++
// DFlash Encoder: processes target model features through feature fusion layer
template <>
llama_model_dflash::graph<true>::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    ggml_tensor * cur = build_inp_embd_enc();

    cur = build_lora_mm(model.fc, cur, model.fc_s);
    cb(cur, "fc_out", -1);

    cur = build_norm(cur, model.output_norm_enc, NULL, LLM_NORM_RMS, -1);
    cb(cur, "enc_norm_out", -1);

    ggml_set_output(cur);
    res->t_h_nextn = cur;

    ggml_build_forward_expand(gf, cur);
}
```
```c++
ggml_tensor * llama_model_dflash::graph<true>::build_inp_embd_enc() const {
    auto inp_target = std::make_unique<llm_graph_input_embd>(hparams.n_embd_inp_enc());

    inp_target->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_inp_enc(), n_tokens);
    ggml_set_input(inp_target->embd);

    ggml_tensor * cur = inp_target->embd;
    cb(cur, "inp_embd", -1);

    res->add_input(std::move(inp_target));

    return cur;
}
```
So later when `process_ubatch` is called the inputs will be set and the following
function will be called:
```c++
void llm_graph_input_embd::set_input(const llama_ubatch * ubatch) {
    if (ubatch->token) {
        const int64_t n_tokens = ubatch->n_tokens;

        ggml_backend_tensor_set(tokens, ubatch->token, 0, n_tokens*ggml_element_size(tokens));
    }

    if (ubatch->embd) {
        GGML_ASSERT(n_embd == embd->ne[0]);

        const int64_t n_tokens = ubatch->n_tokens;

        ggml_backend_tensor_set(embd, ubatch->embd, 0, n_tokens*n_embd*ggml_element_size(embd));
    }
}
```
And this will set the embeddings by copying from the ubatch to the backend.
And notice that the encoder is pretty simple, it passes the input embeddings which
will be in cur to:
```c++
    // pass the fully connected tensor which is one that will use the full connected
    // fc tensor (merging the features and down projecting)
    // and the scale if there is a tensor level scale factor.
    cur = build_lora_mm(model.fc, cur, model.fc_s);

    cur = build_norm(cur, model.output_norm_enc, NULL, LLM_NORM_RMS, -1);

    ggml_set_output(cur);
    res->t_h_nextn = cur;
```
The learned fc.weight matrix projects each 16128-value vector down to the DFlash
hidden size:
```console
cur has the shape: [16128, n_tokens]

[16128 target features]
            |
            | fc.weight
            v
  [2688 DFlash features]
```
For every token, this operation:
* combines information from all six selected target layers;
* learns which parts of those layers are useful;
* reduces the feature size from 16128 to 2688;
* produces a vector that the DFlash decoder can use.

