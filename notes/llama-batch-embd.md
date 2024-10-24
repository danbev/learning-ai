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
model should be the context unaware embeddings. I initially thought that these
embeddings could be the result of a program like `llama_embedding` which can
output context aware embeddings. I actually tried this but was not able to get
it to work and while I decided to write this document to sort out my thoughts.

_wip_
