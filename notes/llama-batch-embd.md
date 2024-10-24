## llama_batch embd
This document will take a look at the `embd` field of `llama_batch` and see how
it can be used and for what purposes.

So this is a field similar to the `llama_batch.token` field which is a pointer
to the input tokens. So we might have a prompt which we tokenize which the
is bacically splitting the prompt into tokens and looking them up in the models
vocabulary. This is what is passed to the `llama_batch.token` field. We can see
in following function that if the batch has tokens then a 1d tensor will be
created for the `inp_tokens`:
```c++
static struct ggml_tensor * llm_build_inp_embd(
        struct ggml_context * ctx,
       struct llama_context & lctx,
        const llama_hparams & hparams,
         const llama_ubatch & batch,
         struct ggml_tensor * tok_embd,
         const llm_build_cb & cb) {
    const int64_t n_embd = hparams.n_embd;

    struct ggml_tensor * inpL;

    if (batch.token) {
        lctx.inp_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, batch.n_tokens);
        cb(lctx.inp_tokens, "inp_tokens", -1);
        ggml_set_input(lctx.inp_tokens);

        inpL = ggml_get_rows(ctx, tok_embd, lctx.inp_tokens);
    } else {
        lctx.inp_embd = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, batch.n_tokens);
        inpL = lctx.inp_embd;
        ggml_set_input(lctx.inp_embd);
    }

    cb(inpL, "inp_embd", -1);

    return inpL;
}
```
After the computation graph has been built the `inp_tokens` tensor will be
populated by the tokens in the batch:
```c++
static void llama_set_inputs(llama_context & lctx, const llama_ubatch & batch) {
    ...

    if (batch.token) {
        const int64_t n_tokens = batch.n_tokens;

        ggml_backend_tensor_set(lctx.inp_tokens, batch.token, 0, n_tokens*ggml_element_size(lctx.inp_tokens));
    }
```
To be clear this tensor is a 1d tensor of tokens and the type is I32, we don't
yet have the token embeddings yet (which are float32 values).
```console
(gdb) p *lctx.inp_tokens
$4 = {type = GGML_TYPE_I32, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x0,
ne = {512, 1, 1, 1}, nb = {4, 2048, 2048, 2048}, op = GGML_OP_NONE,
op_params = {0 <repeats 16 times>}, flags = 0, grad = 0x0, src = {0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
0x0, 0x0, 0x0}, view_src = 0x0, view_offs = 0, data = 0x0, name = '\000' <repeats 63 times>, extra = 0x0}
```
The token embeddings can be thought of being looked up from the model which has
learned an embedding representation of the tokens in its vocabulary. These
embeddings don't have any context yet, so the token embedding for the token
`cold` will be the same regardless of the context (sentence) in which it is used.
Notice that `inpL` is then assigned to the tensor operation `ggml_get_rows`
which will use the tokens as indices to look up the embeddings in the `tok_embd`
tensor which is a 2d tensor of float32 (or quantized type but not I32 is my main
point here) values:
```console
(gdb) p *tok_embd
$7 = {type = GGML_TYPE_Q4_K, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x555555a0cf10,
ne = {4096, 32000, 1, 1}, nb = {144, 2304, 73728000, 73728000},
op = GGML_OP_NONE, op_params = {0 <repeats 16 times>}, flags = 0, grad = 0x0, src = {0x0,
0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0}, view_src = 0x0, view_offs = 0, data = 0x7fff042b4ec0,
name = "token_embd.weight", '\000' <repeats 46 times>, extra = 0x0}
```
So this is how the lookup is performed.

Now, with that in mind when we have `batch.embd` set to point to embeddings in
stead of having tokenss in the batch the `inp_embd` tensor will be created
directly as a 2d tensor of float32 values:
```c++
        lctx.inp_embd = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, batch.n_tokens);
        inpL = lctx.inp_embd;
        ggml_set_input(lctx.inp_embd);
```
And later in the `llama_set_inputs` the `inp_embd` tensor will be populated
with the embeddings from the batch:
```c++
static void llama_set_inputs(llama_context & lctx, const llama_ubatch & batch) {
    ...
    if (batch.embd) {
        const int64_t n_embd   = hparams.n_embd;
        const int64_t n_tokens = batch.n_tokens;

        ggml_backend_tensor_set(lctx.inp_embd, batch.embd, 0, n_tokens*n_embd*ggml_element_size(lctx.inp_embd));
    }
```
So my understanding of `batch.embd` is that the embeddings we can pass to the
model should be the context _unaware_ embeddings.

I initially thought that these embeddings could be the result of a program like
`llama_embedding` which can output context aware embeddings. I actually tried
this but was not able to get it to work and while I decided to write this
document to sort out my thoughts.

In that case I was trying to use context aware embeddings, so they had already
been processed by the llama.cpp. But as we saw above these would then be used
as inputs to the model and go through all the layers of the model (with the
self-attention and feed-forward layers) and then we would have predict the
next token based on those embeddings. 

This is what I did when testing:

First I created embeddings using `llama-embedding` just to be able to make sure
that the same embeddings are generated by the example (I'll show/link to it
shortly):
```console
./llama-embedding -m /home/danbev/work/ai/learning-ai/fundamentals/llama.cpp/models/llama-2-7b-chat.Q4_K_M.gguf --pooling none -p "What is LoRA?" --embd-normalize -1 --verbose-prompt
...
main: prompt 0: 'What is LoRA?'
main: number of tokens in prompt = 6
     1 -> '<s>'
  1724 -> ' What'
   338 -> ' is'
  4309 -> ' Lo'
  4717 -> 'RA'
 29973 -> '?'

batch_decode: n_tokens = 6, n_seq = 1

embedding 0:  0.400597 -0.190815  0.222652  ... -0.070679  0.036515  0.249583
embedding 1:  1.491324  0.126824  1.260321  ... -0.386189 -1.460397  1.050678
embedding 2:  0.130336 -0.622851  2.395198  ... -1.197654 -0.593481 -0.097434
embedding 3: -0.708625 -0.107112 -0.749924  ...  2.177819 -4.191534 -0.090702
embedding 4: -1.278546  0.123264 -1.812276  ... -1.009741  2.422771  0.410338
embedding 5:  1.063118 -2.252861 -2.212228  ... -0.188066 -0.433652  0.414811
```

Then in the example [embeddings](../fundamentals/llama.cpp/src/embeddings.cpp)
I did the same thing and extracted the embeddings:
```
embedding 0 0.400597 -0.190815 0.222652 0.074640 -0.191031
embedding 1 1.491324 0.126824 1.260321 -0.551602 0.151728
embedding 2 0.130336 -0.622851 2.395198 1.474568 -1.747949
embedding 3 -0.708625 -0.107112 -0.749924 0.901563 -1.643292
embedding 4 -1.278546 0.123264 -1.812276 -0.099625 -3.626511
embedding 5 1.063118 -2.252861 -2.212228 0.937880 -4.282171
```
So that looks pretty good and these are stored in `token_embeddings`.

Now, I was not sure if I should use the last token embedding or all of them for
inference (not that like I mentioned above this might be completely wrong and
not supposed to work like this at all but I'm including this to clarify what I'm
doing).

_wip_
