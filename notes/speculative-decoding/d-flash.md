## D-Flash speculative decoding
What D-Flash addresses is the fact that when we have a draft model it is still
an autoregressive model, like medusa and eagle,  which processes one token at a
time which means that there is a limit to have good these types of models can
perform. The D in D-Flash stands for diffusion.

So diffusion will generate all the "stuff" at once if for example we think of
an image. But in an LLM we don't have a fixed size output to generate which is
a requirement of a diffusion model (if I recall correctly). So what D-Flash does
is it creates blocks of tokens of a fixed size and then it generates those
blocks in one forward process or the diffusion model.

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
So initially this would just be noice, like in stable diffusion where we start
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
So this happens in the speculative draft step, which is part of
common/speculative.cpp:
```c++
    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {
        const int n_embd           = llama_model_n_embd(llama_get_model(ctx_dft_dec));
        // block_size is bounded by the model's trained block_size (from GGUF metadata).
        const int model_block_size = llama_model_dflash_block_size(llama_get_model(ctx_dft_dec));
        const int block_size       = std::min((int)params.n_max, model_block_size);
        const int n                = (int)prompt_tgt.size();
        const int n_new            = n - dflash_n_past;

        GGML_ASSERT(n >= 1 && "prompt_tgt is empty");
        GGML_ASSERT(n_new >= 1 && "must have at least 1 new token");

        // Just like Eagle 3 take more than the final layer from the target model
        // D-Flash also picks multiple target layer output. So this is getting
        // them all for the last token.
        const float * features = llama_get_dflash_target_features(ctx_tgt);

        llama_batch enc_batch = {
            /*.n_tokens  =*/ n_new,
            /*.token     =*/ nullptr,
            /*.embd      =*/ const_cast<float*>(features),
            /*.pos       =*/ nullptr,
            /*.n_seq_id  =*/ nullptr,
            /*.seq_id    =*/ nullptr,
            /*.logits    =*/ nullptr,
        };
        // We then _encode_ these features. The draft model is much smaller than the target
        // model so this is like a down projection using model.fc. We might have picked
        // 3 layers from the target model so we need to shrink them down. But it might
        // also be doing feature translation and other things as this is a learned translation.
        if (llama_encode(ctx_dft_enc, enc_batch) != 0) {
            LOG_ERR("DFlash: encoder failed\n");
            return;
        }

        // Then we get the output of the encoder:
        const float * target_ctx_new = llama_get_embeddings(ctx_dft_enc);
        GGML_ASSERT(target_ctx_new && "encoder output is null");

        // Then we append these to the accumulated_ctx vector as a range. So this
        // will grow larger and larger we go.
        const size_t new_size = (size_t)n_embd * n_new;
        accumulated_ctx.insert(accumulated_ctx.end(), target_ctx_new, target_ctx_new + new_size);

        const int n_ctx_total = (int)(accumulated_ctx.size() / n_embd);
        llama_set_dflash_accumulated_target_ctx(ctx_dft_dec, accumulated_ctx.data(), n_embd, n_ctx_total);
```
And this will be copied into the llama_context's cross member:
```c++
void llama_context::set_dflash_accumulated_target_ctx(const float * data, int32_t n_embd, int32_t n_tokens) {
    GGML_ASSERT(data != nullptr);
    const size_t size = (size_t)n_embd * n_tokens;
    // Store in cross struct (reusing T5 style cross-attention for accumulated target features fed to the DFlash decoder)
    cross.n_embd = n_embd;
    cross.n_enc  = n_tokens; // n_ctx_total
    cross.v_embd.resize(size);
    std::memcpy(cross.v_embd.data(), data, size * sizeof(float));
}
```
```c++
struct llama_cross {
    int64_t n_embd = 0;
    int64_t n_enc  = 0;

    // embeddings data copied to host memory (tmp)
    std::vector<float> v_embd;

    // needed to construct the cross-attention mask in the decoder
    std::vector<std::set<llama_seq_id>> seq_ids_enc;
};
```

Following that we have:
```c++
        // This is the noice token id
        const llama_token mask_token_id = llama_model_dflash_mask_token_id(llama_get_model(ctx_dft_dec));

        // And we will now clear the batch and add the last llama_token id (which is passed
        // into this draft function. The remaining tokens will get the mask_token_id.
        common_batch_clear(batch);
        for (int i = 0; i < block_size; i++) {
            const llama_token tok = (i == 0) ? id_last : mask_token_id;
            common_batch_add(batch, tok, i, {0}, true);
        }

        // Then we call decode which is the Diffusion part.
        if (llama_decode(ctx_dft_dec, batch) != 0) {
            LOG_ERR("DFlash: noise decode failed\n");
            return;
        }

        dflash_n_past = n;

        // And then we sample from the result of the diffusion step.
        result.clear();
        common_sampler_reset(smpl);

        for (int i = 1; i < block_size; i++) {
            common_sampler_sample(smpl, ctx_dft_dec, i);

            const auto * cur_p = common_sampler_get_candidates(smpl, true);
            const llama_token id = cur_p->data[0].id;

            common_sampler_accept(smpl, id, true);
            result.push_back(id);
        }
    }
```
So just to be really clear about this. The first time draft is called we have
already processed the initial prompt, so we will always have target model features
to use, they will be from the initial prompt the first time. And after that, for
token generation, we will be using the drafting process.
So the target model will handle the initial decode and since it need to store
the target feature layer outputs the target model needs to know about dflash.
```c++
llm_build_qwen3::llm_build_qwen3(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    ...
    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;
        ...
        // DFlash: Extract intermediate layer features from target model at layer INPUT
        if (dflash && cparams.dflash_extract_enabled && !dflash->extract_layer_indices.empty()) {
            static const char * dflash_extract_names[] = {
                "dflash_extract_0", "dflash_extract_1", "dflash_extract_2",
                "dflash_extract_3", "dflash_extract_4"
            };
            for (size_t i = 0; i < dflash->extract_layer_indices.size() && i < 5; ++i) {
                if (dflash->extract_layer_indices[i] == il) {
                    cb(inpL, dflash_extract_names[i], il);
                    break;
                }
            }
        }
        ...
    }
    ...
}
```
If we look in llama-context.cpp we have:
```c++
llm_graph_cb llama_context::graph_get_cb() const {
    return [&](const llama_ubatch & ubatch, ggml_tensor * cur, const char * name, int il) {
        if (il >= 0) {
            ggml_format_name(cur, "%s-%d", name, il);
        } else {
            ggml_set_name(cur, name);
        }
        ...
        // DFlash: Extract intermediate layer features if this is an extraction point
        if (cparams.dflash_extract_enabled) {
            static constexpr const char * prefix = "dflash_extract_";
            static constexpr size_t prefix_len = 15;

            if (strncmp(name, prefix, prefix_len) == 0) {
                size_t extract_idx = 0;
                if (sscanf(name + prefix_len, "%zu", &extract_idx) == 1 && extract_idx < dflash.extract_tensors.size()) {
                    ggml_set_output(cur);
                    dflash.extract_tensors[extract_idx] = cur;
                }
            }
        }
        ...

```
So the callback will set the tensors as output, and will also store them in
the dflash member of llama-context:
```c++
struct llama_context {
    ...
    mutable llama_dflash dflash;
```
```c++
// DFlash intermediate results struct (similar to Eagle3)
struct llama_dflash {
    std::vector<int> extract_layer_indices;

    std::vector<float> target_features;

    std::vector<ggml_tensor *> extract_tensors;

    void clear() {
        target_features.clear();
        extract_tensors.clear();
    }
};
```
So that is how the target model extract the feature layers.

Now, lets turn our attention to the draft models decode function (the encoder
is very simple so I'm skipping it for now.
```c++
llm_build_dflash_decode::llm_build_dflash_decode(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    // Noise tokens [MASK]
    GGML_ASSERT(model.target_tok_embd != nullptr && "DFlash decoder requires target model's tok_embd");
    ggml_tensor * noise_embd = build_inp_embd(model.target_tok_embd);
    cb(noise_embd, "inp_noise_embd", -1);
```
Recall that before we call decode we created a batch which contained the last
token id and the rest were the mask token id. This is just treated as a normal
input of token ids, so they will be looked up using ggml_get_rows in the model.

Next we have the cross attention tensor that is create as an input tensor, which
will later be filled with the features from the target model:
```c++
    // Target context via llama_cross (filled from accumulated_target_ctx), graph rebuilds every step
    ggml_tensor * target_ctx = build_inp_cross_embd();
    const int64_t n_ctx = target_ctx->ne[1];
```
So I was expecting to find some diffusion "magic" for a lack of a better word but
that is part of how the model was trained, it was trained to denoise a corrupted
input sequence. Here the noice is what we introduced with the mask_token_id's.
The actual graph uses cross attention, where the features from the target model
enable the model to predict good outputs.
