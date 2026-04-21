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

        const float * target_ctx_new = llama_get_embeddings(ctx_dft_enc);
        GGML_ASSERT(target_ctx_new && "encoder output is null");

        // Step 2: Append to accumulated target_ctx and set on decoder context (writes to cross.v_embd)
        const size_t new_size = (size_t)n_embd * n_new;
        accumulated_ctx.insert(accumulated_ctx.end(), target_ctx_new, target_ctx_new + new_size);

        const int n_ctx_total = (int)(accumulated_ctx.size() / n_embd);
        llama_set_dflash_accumulated_target_ctx(ctx_dft_dec, accumulated_ctx.data(), n_embd, n_ctx_total);

        // Step 3: Decode noise block
        const llama_token mask_token_id = llama_model_dflash_mask_token_id(llama_get_model(ctx_dft_dec));

        common_batch_clear(batch);
        for (int i = 0; i < block_size; i++) {
            const llama_token tok = (i == 0) ? id_last : mask_token_id;
            common_batch_add(batch, tok, i, {0}, true);
        }

        if (llama_decode(ctx_dft_dec, batch) != 0) {
            LOG_ERR("DFlash: noise decode failed\n");
            return;
        }

        dflash_n_past = n;

        // Step 4: Sample draft tokens from positions 1..block_size-1
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
