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
the complete input embeddings and then populate the encoders output state (the
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

_wip_
