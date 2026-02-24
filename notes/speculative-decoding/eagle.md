## Eagle speculative decoding (Extrapolation Algorithm for Greater Language-model Efficiency)
Framework for Lightweight Autoregressive Decoding.

### Eagle 1
Similar to Medusa, this technique does not require a separate draft model, but
instead adds a small head to the main model.

This is autoregessive, but the standard speculative decoding approach is also
autoregressive so how is this different?  
Standard speculative decoding is just like a smaller model, that predicts the
next token. It works in "token" space so it outputs token ids. Eagle 1 changes
this and instead predicts the next hidden state for the next token.

So compared to [Medusa](./medusa.md) which has separate lm_heads for each
token that is predicted Eagle only has one lm_head. The process looks something
like the following:
```
            main_model: predicts h_t
                           |
            eagle heads(N) +--------------------------------+
                           ↓                                |
                        lm_head(h_t) -> l_t     (logits)    |
                           ↓                                |
                        sample(l_t) -> x_{t+1}  (token id)  |
                           ↓                                |
                        inp_embd = tok_emb(x_{t+1})         |
                           ↓                                |
                        inp_eagle = concat(h_t, inp_embd)   |
                           ↓                                |
                        eagle layer predicts h_{t+1}        |
                           ↓                                |
                           ------------h_{t+1}--------------+
                           |    prediction table:
                           |    x{t+1} : l_t[x_{t+1}]
                           |    x{t+2} : l_t[x_{t+2}]
                           ↓    x{t+3} : l_t[x_{t+3}]
                        batch
                           |tokens[:
                           |  prompt
                           |  x_{t+1}
                           |  x_{t+2}
                           |  x_{t+3}
                           |]
                           ↓
           main_model: processes batch and verify
```
Above is just a simple diagram to try to get an overview of the process. One
major difference is that Eagle 1 uses a static tree structure and often predicts
more than a single token. For example for each iteration of the eagle head we
might do N predictions.

Notice that h_t and inp_embd are actually concatenated together so this will be
a larger vector that the original hidden space.

```
Tree Key:
k = 2
( ) = Hidden State [Vector]
[ ] = Token ID [Sampled]

           (h_t) <-- Main model prediction
             |
      +------+------+
      ↓             ↓
    [cat]         [the]    <-- Top-2 Tokens (x_{t+1})
      |             |
    (h_t+1)       (h_t+1') <-- Eagle Predictions
      |             |
   +--+--+       +--+--+
   ↓     ↓       ↓     ↓
 [sat] [ran]   [end] [sun] <-- Top-2 for x_{t+2}
```
Instead of a single chain we now have an array with this tree. Now if the
prediction is "cat->sat", but the real model wanted "cat-ran" the we would loose
the second prediction. With a tree structure we can keep the "ran" prediction:
```
[cat, dog, sat, ran, end, sun]
```

```
Draft chain: ["The", "cat", "sat", "on"]
Real Models: ["The", "cat", "slept", "..."]
```
Here "sat" is rejected so the rest of the draft model predictions are also
rejected. BUt perhaps the real model would have accepted "on" after "slept" but
with the simple chain we would never get to see that prediction.

Also the tree is not just a simple list, each node contains the cumulative log
probability of the token that was predicted. So it is easy to add to the tree,
we just add the log(probability) of the new token to the cumulative log
probability of the parent.


### Eagle 2
Is very similar to Eagle 1 but the token tree is not static but dynamic and can
be pruned if the cumulative log probability of a branch is too low.

TODO: expand on this.

### Eagle 3
So if we think about the two prior version above we saw that the take the
predicted hidden state h_t and the concatenate it with the token embedding of
the predicted token. In Eagle3 instead of taking just the models output (h_t)
they also take a hidden state from the lower level, a mid level, and a later
level (might be the last but I'm not sure yet).

For example, there is a "extract_layers" parameter that specifies which hidden
states to take:
```console
(venv) spark $ gguf-dump models/EAGLE3-LLaMA3.1-Instruct-8B_fp16.gguf
INFO:gguf-dump:* Loading: models/EAGLE3-LLaMA3.1-Instruct-8B_fp16.gguf
* File is LITTLE endian, script is running on a LITTLE endian host.
* Dumping 37 key/value pair(s)
      1: UINT32     |        1 | GGUF.version = 3
      2: UINT64     |        1 | GGUF.tensor_count = 14
      3: UINT64     |        1 | GGUF.kv_count = 34
      4: STRING     |        1 | general.architecture = 'eagle3'
      5: [INT32]    |        3 | eagle3.extract_layers = [2, 16, 29]
```
And this is set in:
```c++
         case LLM_ARCH_EAGLE3:
             {
                 ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                 // EAGLE3 layer extraction configuration
                 // Use array<int, 4> (has template instantiation), then copy first 3 elements
                 std::array<int, 4> extract_layers_tmp = {};
                 if (!ml.get_key_or_arr(LLM_KV_EAGLE3_EXTRACT_LAYERS, extract_layers_tmp, 3, false)) {
                     throw std::runtime_error("EAGLE3 model requires 'extract_layers' in GGUF metadata");
                 }
                 std::copy_n(extract_layers_tmp.begin(), 3, hparams.eagle3_extract_layers.begin());
                 LLAMA_LOG_INFO("%s: EAGLE3 extract_layers = [%d, %d, %d]\n", __func__,
                                hparams.eagle3_extract_layers[0],
                                hparams.eagle3_extract_layers[1],
                                hparams.eagle3_extract_layers[2]);
```

Now, if they are concatenating all of these together then the input this will be
very large and unpractical. What is done instead is these are merged/fused
together. This is done using a separate encoder in llama.cpp. Then we have the
decoder which is the actual speculator itself, the one that generates draft
tokens.

But lets start by looking at how a concrete example works to get a better
picture of the whole process. I'll be looking at speculative-simple for this:
```console
#!/bin/bash

set -e

build_dir=build-dgx-spark-debug
model=models/Llama-3.1-8B-Instruct_bf16.gguf
md=models/EAGLE3-LLaMA3.1-Instruct-8B_fp16.gguf
cmd=llama-speculative-simple

cmake --build $build_dir --target $cmd -j 12

gdb --args ${build_dir}/bin/$cmd \
      -m $model \
      -md $md \
      --eagle3 \
      -p "What is the capital of France?" \
      -n 20 \
      --draft 8 \
      --temp 0 --top-k 1 --seed 18 -ngl 99 -ngld 99
```
The conversion for this was done using this script:
```console
#!/bin/bash

set -e

source venv/bin/activate

#mkdir -p /home/danbev/work/models/meta
#pushd /home/danbev/work/models/meta
#hf download meta-llama/Llama-3.1-8B-Instruct --local-dir Llama-3.1-8B-Instruct
#popd

TARGET_MODEL_HF="/home/danbev/work/models/meta/Llama-3.1-8B-Instruct"
TARGET_MODEL_GGUF="./models/Llama-3.1-8B-Instruct_bf16.gguf"

python convert_hf_to_gguf.py \
    "${TARGET_MODEL_HF}" \
    --outtype bf16 \
    --outfile "${TARGET_MODEL_GGUF}"

#mkdir -p /home/danbev/work/models/yuhuili
#pushd /home/danbev/work/models/yuhuili
#hf download yuhuili/EAGLE3-LLaMA3.1-Instruct-8B --local-dir EAGLE3-LLaMA3.1-Instruct-8B
#popd

EAGLE3_MODEL_HF="/home/danbev/work/models/yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"
EAGLE3_MODEL_GGUF="./models/EAGLE3-LLaMA3.1-Instruct-8B_fp16.gguf"

python convert_hf_to_gguf.py \
    "${EAGLE3_MODEL_HF}" \
    --outtype f16 \
    --target-model-dir "${TARGET_MODEL_HF}" \
    --outfile "${EAGLE3_MODEL_GGUF}"

deactivate
```
Setting a breakpoint in speculative-simple's main function we can start by
looking at the decoding of the initial propt which is done by the main model:
```c++ 
if (params.speculative.eagle3) {
    // Target model decodes full prompt and sample first token and intermediate features are extracted
    llama_decode(ctx_tgt, llama_batch_get_one(inp.data(), inp.size()));

    id_last = common_sampler_sample(smpl, ctx_tgt, -1);
    common_sampler_accept(smpl, id_last, true);
    LOG("%s", common_token_to_piece(ctx_tgt, id_last).c_str());
    n_predict++;

    // all tokens currently in the target context
    prompt_tgt.assign(inp.begin(), inp.end());
    prompt_tgt.reserve(llama_n_ctx(ctx_tgt));

    n_past = inp.size();
} else {
```
Notice that this is using the main model (ctx_tgt) (target model):
```console
(gdb) p ctx_tgt->model->name
$3 = "Llama 3.1 8B Instruct"
```
So this is just a normal decoding of the prompt but there is something else
happing in `llama_context::process_ubatch`.
```c++
llm_graph_result * llama_context::process_ubatch(const llama_ubatch & ubatch,
        llm_graph_type gtype, ll     ama_memory_context_i * mctx, ggml_status & ret) {
    ...
    // EAGLE3: Extract intermediate layer features after graph execution
    if (cparams.eagle3_extract_enabled && !eagle3.extract_tensors.empty()) {
        extract_eagle3_features(ubatch);
    }

    ret = GGML_STATUS_SUCCESS;

    return res;
}
```
```c++
void llama_context::extract_eagle3_features(const llama_ubatch & ubatch) {
    const int64_t n_tokens = ubatch.n_tokens;
    const int64_t n_embd = model.hparams.n_embd;
    const size_t n_layers = eagle3.extract_tensors.size();
```
```console
(gdb) p n_tokens
$4 = 50
(gdb) p n_embd
$5 = 4096
(gdb) p n_layers
$6 = 3
```
```c++
    const int64_t n_embd_concat = n_embd * n_layers;
    eagle3.target_features.resize(n_embd_concat * n_tokens);
```
```console
(gdb) p eagle3.target_features.size()
$11 = 614400
```
Following that we have: 
```c++
    // Temporary buffer to hold layer features before transposing
    static thread_local std::vector<float> temp_layer_features;
    temp_layer_features.resize(n_embd * n_tokens);
```
Notice the use of thread local here and that this is static. So the first time
a thread enters this function it will create a new std::vector in its thread
local storage area. And this is destroyed when the thread exits.
Next we loop over all the 3 layers (low, mid, high) that we mentioned above:
```c++
    for (size_t layer_idx = 0; layer_idx < n_layers; ++layer_idx) {
        ggml_tensor * tensor = eagle3.extract_tensors[layer_idx];
```
```console
(gdb) p tensor->name
$15 = "eagle3_extract_0-2", '\000' <repeats 45 times>
(gdb) p tensor->ne
$16 = {4096, 50, 1, 1}
```
Perhaps this could use ggml_nbytes instead:
```console
(gdb) p size_bytes
$17 = 819200
(gdb) p ggml_nbytes(tensor)
$18 = 819200
```
Next the backend for the tensor is retreived and then we get the data from the
tensor and write it into the temp_layer_features buffer:
```c++
        const size_t size_bytes = n_embd * n_tokens * sizeof(float);
        ggml_backend_tensor_get_async(backend, tensor, temp_layer_features.data(), 0, size_bytes);
        ggml_backend_sched_synchronize(sched.get());
```
Notice that we synchronize after this call so that we are sure that the data
is copied.
Following that we are going to copy this data into the target_features vector:
```c++
        for (int64_t token_idx = 0; token_idx < n_tokens; ++token_idx) {
            const float * src = temp_layer_features.data() + token_idx * n_embd;
            float * dest = eagle3.target_features.data() + token_idx * n_embd_concat + layer_idx * n_embd;
            std::memcpy(dest, src, n_embd * sizeof(float));
        }
```
So after process_ubatch has run the eagle3 member of llama_context has had its
target_features vector filled with the extracted features from the main model.

Next in speculative-simple we have:
```c++
    const auto & params_spec = params.speculative;

    struct common_speculative * spec = common_speculative_init(params.speculative, ctx_tgt);

    common_speculative_begin(spec, prompt_tgt);
```
The eagle3 begin function is a noop.

```c++
        llama_tokens draft = common_speculative_draft(spec, params_spec, prompt_tgt, id_last);

```
This will call into common_speculative_state_eagle3::draft:
```c++
    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {
        auto * spec = this;

        auto & batch       = spec->batch;
        auto & ctx_tgt     = spec->ctx_tgt;
        auto & ctx_dft_enc = spec->ctx_dft_enc;
        auto & ctx_dft_dec = spec->ctx_dft_dec;
        auto & smpl        = spec->smpl;
    }
```
```console
(gdb) p ctx_dft_enc.model->name
$26 = "EAGLE3 LLaMA3.1 Instruct 8B"
(gdb) p ctx_dft_dec.model->name
$27 = "EAGLE3 LLaMA3.1 Instruct 8B
(gdb) p ctx_dft_dec.model->layers.size()
$28 = 1
(gdb) p ctx_dft_enc.model->tensors_by_name
$30 = std::vector of length 14, capacity 16 = {
{first = "d2t", second = 0xaaaab8a3a6a0},
{first = "fc.weight", second = 0xaaaab8a3bf20},
{first = "output_norm.weight", second = 0xaaaab8a3c090},
{first = "output.weight", second = 0xaaaab8a3c200},
{first = "blk.0.attn_norm.weight", second = 0xaaaab8a3c370},
{first = "blk.0.attn_q.weight", second = 0xaaaab8a3c4e0},
{first = "blk.0.attn_k.weight", second = 0xaaaab8a3c650},
{first = "blk.0.attn_v.weight", second = 0xaaaab8a3c7c0},
{first = "blk.0.attn_output.weight", second = 0xaaaab8a3c930},
{first = "blk.0.hidden_norm.weight", second = 0xaaaab8a3caa0},
{first = "blk.0.ffn_norm.weight", second = 0xaaaab8a3cc10},
{first = "blk.0.ffn_gate.weight", second = 0xaaaab8a3cd80},
{first = "blk.0.ffn_down.weight", second = 0xaaaab8a3cef0},
{first = "blk.0.ffn_up.weight", second = 0xaaaab8a3d060}}
```

```c++
        // Clear draft positions from decoder KV cache [n_past, inf)
        llama_memory_seq_rm(llama_get_memory(ctx_dft_dec), 0, spec->eagle3_n_past, -1);
```
```
        const float * features = llama_get_eagle3_target_features(ctx_tgt);
```
This call will be delegated to llama_context::get_eagle3_target_features:
```c++
const float * llama_context::get_eagle3_target_features() const {
    GGML_ASSERT(!eagle3.target_features.empty() && "EAGLE3 target features not extracted - call llama_encode() on target model first");
    return eagle3.target_features.data();
}
```
And this is the same vector that we saw was filled above in process_ubatch.
The we will create a new batch but with the embeddings field set to the
extracted features:
```c++
        llama_batch enc_batch = {
            /*.n_tokens  =*/ n_new,
            /*.token     =*/ nullptr,
            /*.embd      =*/ const_cast<float*>(features),
            /*.pos       =*/ nullptr,
            /*.n_seq_id  =*/ nullptr,
            /*.seq_id    =*/ nullptr,
            /*.logits    =*/ nullptr,
        };
        GGML_ASSERT(llama_encode(ctx_dft_enc, enc_batch) == 0);
```
And this is also sent to llama_encode. And this will run the encoder part of the
draft model which recall is bidirectional and non-casual. This will process
the three hidden states that we populated by the target model by the first
decode that we saw previously. The encoder will take the three hidden states and
run a bidirectional attention over them which produces a fused representation.

cross attention cache) which the decoder will then attend to. So calling the
encoder is so that we can merge/fuse the features together like we mentioned
above and then the decoder can attend to this fused representation.
Because it's bidirectional, each position can attend to all others across all
three feature levels simultaneously.

```c++
int llama_context::encode(const llama_batch & batch_inp) {
    GGML_ASSERT((!batch_inp.token && batch_inp.embd) || (batch_inp.token && !batch_inp.embd)); // NOLINT

    if (batch_inp.n_tokens == 0) {
        LLAMA_LOG_ERROR("%s: n_tokens == 0\n", __func__);
        return -1;
    }

    const auto & hparams = model.hparams;

    // EAGLE3: use 3*target_hidden_size for concatenated features input
    const int64_t n_embd  = (model.arch == LLM_ARCH_EAGLE3 && batch_inp.embd) ? 3 * hparams.eagle3_target_hidden_size : hparams.n_embd;
    const int64_t n_vocab = model.vocab.n_tokens();
```
Lets take closer look at the graph for the encoder:
```c++
llm_build_eagle3_encode::llm_build_eagle3_encode(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    ggml_tensor * cur = nullptr;

    cur = build_inp_embd();

    // Feature fusion layer
    cur = build_lora_mm(model.fc, cur);
    cb(cur, "fc_out", -1);

    // Output: g_embeddings e.g. [4096, n_tokens]
    res->t_embd = cur;

    ggml_build_forward_expand(gf, cur);
}
```
So the encoder is very simple it will 
```console
(gdb) p cur->ne
$2 = {12288, 1, 1, 1}

(gdb) p model.fc->ne
$4 = {12288, 4096, 1, 1}

(gdb) p cur->ne
$3 = {4096, 1, 1, 1}
```
And notice that the output is set to res->t_embd. If we look back in llama_encode
we can see how this is used:
```c++
    const uint32_t n_embd_out = hparams.n_embd_out();

    if (model.arch == LLM_ARCH_EAGLE3) {
        // g_embeddings are stored temporarily in embd buffer
        const int64_t out_embd = hparams.n_embd;
        GGML_ASSERT(n_tokens * out_embd <= (int64_t) embd.size);
        ggml_backend_tensor_get_async(backend_embd, t_embd, embd.data, 0, n_tokens * out_embd * sizeof(float));
    } else {
```
So this is setting the `embd` buffer, which is a field in llama-context. Now,
next in speculative.cpp we have:
```c++
        // just showing llama_encode for context but we have already looked at this above
        GGML_ASSERT(llama_encode(ctx_dft_enc, enc_batch) == 0);

        const float * g_embd = llama_get_embeddings(ctx_dft_enc);

        // Decoder batch: process new tokens with KV cache reuse
        llama_set_eagle3_g_embeddings(ctx_dft_dec, g_embd, n_embd, n_new);
```
So this is getting a pointer to embd data:
```c++
float * llama_context::get_embeddings() {
    output_reorder();

    return embd.data;
}
```
And this is used to in the call to llama_set_eagle3_g_embeddings:
```c++
void llama_set_eagle3_g_embeddings(llama_context * ctx, const float * g_embd, int32_t n_embd, int32_t n_tokens) {
    ctx->set_eagle3_g_embeddings(g_embd, n_embd, n_tokens);
}

void llama_context::set_eagle3_g_embeddings(const float * g_embd, int32_t n_embd, int32_t n_tokens) {
    GGML_ASSERT(g_embd != nullptr && "g_embeddings cannot be null");
    GGML_ASSERT(n_embd > 0 && n_tokens > 0 && "invalid dimensions");

    const size_t size = n_embd * n_tokens;
    eagle3.g_embeddings.resize(size);
    std::memcpy(eagle3.g_embeddings.data(), g_embd, size * sizeof(float));
}
```
So after this method returns we ahve set the g_embeddings field of the eagle3
member of llama_context.
Next the batch is cleared and a new batch is prepared:
```c++
        common_batch_clear(batch);
        for (int i = 0; i < n_new; i++) {
            const int pos = spec->eagle3_n_past + i;
            const llama_token tok = (pos < n - 1) ? prompt_tgt[pos + 1] : id_last;
            common_batch_add(batch, tok, pos, {0}, true);
        }
```
```console
(gdb) p n_new
$6 = 50
```
And the `id_last` came from speculative-simple.cpp and is just the last of the
input tokens:
```console
gdb) up
#2  0x0000aaaaaab641d8 in main (argc=22, argv=0xffffffffef98)
    at /home/danbev/work/llama.cpp-work/examples/speculative-simple/speculative-simple.cpp:200
200	        llama_tokens draft = common_speculative_draft(spec, params_spec, prompt_tgt, id_last);
(gdb) f
#2  0x0000aaaaaab641d8 in main (argc=22, argv=0xffffffffef98)
    at /home/danbev/work/llama.cpp-work/examples/speculative-simple/speculative-simple.cpp:200
200	        llama_tokens draft = common_speculative_draft(spec, params_spec, prompt_tgt, id_last);
```
And then this batch will be decoded by the draft models decoder:
```c++
        GGML_ASSERT(llama_decode(ctx_dft_dec, batch) == 0);
```
Notice that the batch only contains the prompt and not the extrated features.

```c++
llm_graph_result * llama_context::process_ubatch(const llama_ubatch & ubatch, llm_graph_type gtype, llama_memory_context_i * mctx, ggml_status & ret) {
    ...
    // set the input data for the input tensors
    {
        //const auto t_start_us = ggml_time_us();

        res->set_inputs(&ubatch);

        // EAGLE3: Fill g_embeddings for decoder input
        if (model.arch == LLM_ARCH_EAGLE3 && gtype == LLM_GRAPH_TYPE_DECODER && !eagle3.g_embeddings.empty()) {
            ggml_tensor * g_embd = ggml_graph_get_tensor(gf, "inp_g_embeddings");
            if (g_embd) {
                ggml_backend_tensor_set(g_embd, eagle3.g_embeddings.data(), 0, ggml_nbytes(g_embd));
            }
        }
        ...
}
```
And above we can see that the inp_g_embeddings tensor is retrieved from the
computation graph. This tensor is created in the build graph function for the
decoder which we will get to shortly but this is how the tensor is currently
created:
```c++
    // TODO: refactor into llm_graph_input
    ggml_tensor * inp_g = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_set_input(inp_g);
    cb(inp_g, "inp_g_embeddings", -1); // TODO: do not change the name! refactor into llm_graph_input
```
And notice that we are using the eagle3.g_embeddings vector to set the data which
set previsously.
And then the graph is computed:
```c++
    const auto status = graph_compute(res->get_gf(), ubatch.n_tokens > 1);
```
So lets take a look at the decoders graph:
```c++
llm_build_eagle3_decode::llm_build_eagle3_decode(const llama_model & model,
    const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v;

    ggml_tensor * token_embd_eagle3 = (model.tok_embd != nullptr) ? model.tok_embd : model.target_tok_embd;
}
```
So the token embedding matrix (which what is used to lookup the token embedding
vectors for tokens) is either retrieved from the model's tok_embd field or the
target_tok_embd field:
```console
(gdb) p model->name
$24 = "EAGLE3 LLaMA3.1 Instruct 8B"
(gdb) p model.tok_embd
$25 = (ggml_tensor *) 0x0
(gdb) p model.target_tok_embd
$26 = (ggml_tensor *) 0xaaaab07f1920
(gdb) p model.target_tok_embd->ne
$27 = {4096, 128256, 1, 1}
(gdb) p model.target_tok_embd->name
$28 = "token_embd.weight", '\000' <repeats 46 times>
```
So I don't think I've seen this type of logit before but my understanding is that
the token embedding tables can be very large. And if the Eagle3 model uses the
same vocab as the target model there is no need to store a duplicate. So by
having a target_tok_embd we can get away with just having the single tensor.

And after that we have:
```c++
    ggml_tensor * inp_embd = build_inp_embd(token_embd_eagle3);
    cb(inp_embd, "inp_embd", -1);
```
And the section that we saw before which creates the `inp_g_embeddings` tensor:
```c++
    ggml_tensor * inp_g = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_set_input(inp_g);
    cb(inp_g, "inp_g_embeddings", -1); // TODO: do not change the name! refactor into llm_graph_input

    inpL = inp_g;

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv();

    const float kq_scale = 1.0f/sqrtf(float(n_embd_head));

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // Single decoder layer (il = 0)
    const int il = 0;
    {
        ggml_tensor * embd_norm = build_norm(inp_embd, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
        cb(embd_norm, "embd_norm", il);

        // Apply hidden_norm to inp_g
        ggml_tensor * g_norm = build_norm(inp_g, model.layers[il].eagle3_hidden_norm, NULL, LLM_NORM_RMS, -1);
        cb(g_norm, "g_norm", il);

        // norm_before_residual: determines what goes into the residual connection (compatible with Readhat eagle3 speculator model)
        // - false (default): use raw inp_g for residual
        // - true: use normalized g_norm for residual
        // inpL is the concatenated input (normalized inp_embd + normalized inp_g)
        ggml_tensor * inpSA = hparams.eagle3_norm_before_residual ? g_norm : inpL;
```
Next we concatenate the normalized input embeddings with the normalized
g_embeddings:
```c++
        cur = ggml_concat(ctx0, embd_norm, g_norm, il);
        cb(cur, "concat_embd", il);
```

```console
(gdb) p embd_norm->ne
$31 = {4096, 1, 1, 1}
(gdb) p g_norm->ne
$32 = {4096, 1, 1, 1}
(gdb) p cur->ne
$33 = {8192, 1, 1, 1}
```
Then we have the self-attention:
```c++
        // Self-attention with concatenated input
        ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
        cb(Qcur, "Qcur", il);

        ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
        cb(Kcur, "Kcur", il);

        ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
        cb(Vcur, "Vcur", il);

        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
        Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

        // rope freq factors, returns nullptr if not available
        ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);

        // RoPE
        Qcur = ggml_rope_ext(
                ctx0, Qcur, inp_pos, rope_factors,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow
                );
        Kcur = ggml_rope_ext(
                ctx0, Kcur, inp_pos, rope_factors,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow
                );

        cb(Qcur, "Qcur_rope", il);
        cb(Kcur, "Kcur_rope", il);

        cur = build_attn(inp_attn,
                model.layers[il].wo, NULL,
                Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
```
Next for the specified output ids we extract those rows from the attention output
and also from the residual connection (so that we can add them together):
```c++
        if (inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        // Add residual and update it
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);
```
Then we have a normalization and feedforward network:
```c++

        // Apply FFN norm to the sum
        cur = build_norm(ffn_inp,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "post_attn_norm", il);

        cur = build_ffn(cur,
                model.layers[il].ffn_up,   NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        // Output norm with residual
        cur = ggml_add(ctx0, cur, ffn_inp);
        cb(cur, "eagle3_prenorm", il);

        inpL = cur;
    }
```
```c++
    cur = inpL;

    // Output prenorm state (for next token's g_embeddings in autoregressive generation)
    ggml_set_output(cur);
    res->t_embd = cur;
```
And here we can see t_embd is set to the cur which is the hidden state before
it is normalized below and the lm_head is applied::
```c++

    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1); cb(cur, "result_norm", -1);

    // lm_head - projects to draft vocabulary
    cur = build_lora_mm(model.output, cur);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
```
So that was only building the graph and we made a detour here as we came across
llama_decode in speculate.cpp and eagle3's draft function:
```c++
        GGML_ASSERT(llama_decode(ctx_dft_dec, batch) == 0);
```
So lets follow along from here:
```c++
        spec->eagle3_n_past = n;  // update verified positions

        // Sample draft tokens
        result.clear();
        common_sampler_reset(smpl);


        auto sample_and_check = [&](int idx) -> bool {
            common_sampler_sample(smpl, ctx_dft_dec, idx);

            const auto * cur_p = common_sampler_get_candidates(smpl, true);
            const llama_token id = cur_p->data[0].id;

            common_sampler_accept(smpl, id, true);
            result.push_back(id);

            return cur_p->data[0].p >= params.p_min;
        };

        if (!sample_and_check(n_new - 1)) {
            return;
        }
```
So this above is going to sample the next token and add it to the results
vector (passed in to this function).
Next, we are going to auto-regressively use the sampled token from above which
is now in the results:
```console
(gdb) p result
$59 = std::vector of length 1, capacity 1 = {6864}

(gdb) p ctx_dft_dec.model.vocab.pimpl->id_to_token[6864]
$62 = {text = "Ġcapital", score = 0, attr = LLAMA_TOKEN_ATTR_NORMAL}

(gdb) p params.n_max
$60 = 8
```
Next we are going to use that sampled token, and the hidden state from the last
token (remember the sampled token was sampled from the last hidden state after
it was normalized and passed through the lm_head to get the logits. We want to
then predict the next token using this predicted token but also the hidden state.
And we also update the g_embeddings to be the hidden state from the last token
so that the next prediction before calling llama_decode:
```c++
        // Autoregressive: use prenorm as g_embd (-1 = last output)
        const float * prenorm = llama_get_embeddings_ith(ctx_dft_dec, -1);

        for (int i = 1; i < params.n_max; i++) {
            GGML_ASSERT(prenorm && "prenorm failed");
            llama_set_eagle3_g_embeddings(ctx_dft_dec, prenorm, n_embd, 1);

            common_batch_clear(batch);
            common_batch_add(batch, result.back(), n - 1 + i, {0}, true);
            GGML_ASSERT(llama_decode(ctx_dft_dec, batch) == 0);

            prenorm = llama_get_embeddings_ith(ctx_dft_dec, -1);

            if (!sample_and_check(0)) {
                break;
            }
        }
```
In this case we will do this for 8 tokens. And notice that the lambda above
has a condition that if the probability of the sampled token is less than p_min
then we will stop.

So back in speculative-simple.cpp we have:
```c++
        llama_tokens draft = common_speculative_draft(spec, params_spec, prompt_tgt, id_last);
```
```console
(gdb) p draft
$64 = std::vector of length 6, capacity 8 = {6864, 315, 9822, 374, 12366, 13}
```
Next, we will clear the batch and then add the last token of the prompt to the
batch followed by the draft tokens:
```c++
        common_batch_clear(batch_tgt);
        common_batch_add  (batch_tgt, id_last, n_past++, { 0 }, true);

        // evaluate the target model on [id_last, draft0, draft1, ..., draftN-1]
        {
            if (draft.size() < (size_t) params_spec.n_min) {
                draft.clear();
            }

            for (size_t i = 0; i < draft.size(); ++i) {
                common_batch_add(batch_tgt, draft[i], n_past + i, { 0 }, true);
            }

            llama_decode(ctx_tgt, batch_tgt);
        }
```
Notice that all the draft tokens are added to the batch and that logits/outputs
are set to true for each. This links into some work related to enabling backend
samplers to be able to handle multiple output for the same sequence. Here the
sequence is the same for all tokens in the batch.

And that batch then decoded by the target model:
```console
(gdb) p ctx_tgt->model->name
$86 = "Llama 3.1 8B Instruct"
```

After the draft tokens have been decoded by the target model, we call the following
function and we pass in the sampler, the target models context and the draft tokens:
```c++
        const auto ids = common_sampler_sample_and_accept_n(smpl, ctx_tgt, draft);
```

```c++
std::vector<llama_token> common_sampler_sample_and_accept_n(struct common_sampler * gsmpl,
    struct llama_context * ctx, const llama_tokens & draft, bool grammar_first) {
    std::vector<int> idxs(draft.size() + 1);
    for (size_t i = 0; i < idxs.size(); ++i) {
        idxs[i] = i;
    }

    return common_sampler_sample_and_accept_n(gsmpl, ctx, idxs, draft, grammar_first);
}
```
So what is with this +1?  
Well, as we will see shortly we are going to sample tokens from the target model
and if all the draft tokens match the sampled tokens then the model has already
done a forward pass for all the draft tokens, which produces a new token. So if
all draft tokens match then we can sample the next token without doing an extra
forward pass.
```console
(gdb) p idxs
$89 = std::vector of length 7, capacity 7 = {0, 1, 2, 3, 4, 5, 6}
```
The actual sampling is done in:
```c++
std::vector<llama_token> common_sampler_sample_and_accept_n(
    struct common_sampler * gsmpl,
    struct llama_context * ctx,
    const std::vector<int> & idxs,
    const llama_tokens & draft,
    bool grammar_first) {

    GGML_ASSERT(idxs.size() == draft.size() + 1 && "idxs.size() must be draft.size() + 1");

    std::vector<llama_token> result;
    result.reserve(idxs.size());

    size_t i = 0;
    for (; i < draft.size(); i++) {
        const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);

        common_sampler_accept(gsmpl, id, true);

        result.push_back(id);

        if (draft[i] != id) {
            break;
        }
    }

    if (i == draft.size()) {
        const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);

        common_sampler_accept(gsmpl, id, true);

        result.push_back(id);
    }

    return result;
}
```

_wip_
