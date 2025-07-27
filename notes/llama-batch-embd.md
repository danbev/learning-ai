## llama_batch embd
This document will take a look at the `embd` field of `llama_batch` and see how
it can be used and for what purposes.

In the llama_batch we have tokens which I understand is a array or int32 values which
are the token id which map to a value in the models vocabulary. These are then
"looked up"/retrieved from the embedding matrix of the model which gives the embedding
vector for each token when processing the batch. But the embd field is a 2d array, the
size of the number of tokens in the batch. Is this for when we somehow have alreay got
the embedding vectors for the tokens and in this case don't need to look them up

### Two Input Modes for `llama_batch`
Mode 1: Token IDs (Most Common)
```
  // Input: Token IDs
  llama_batch batch;
  batch.token = [123, 456, 789];  // Token IDs
  batch.embd = nullptr;           // No embeddings provided

  // Model does: embedding_matrix[token_id] → embedding_vector
````

Mode 2: Direct Embeddings (Special Cases)
```
  // Input: Pre-computed embeddings
  llama_batch batch;
  batch.token = nullptr;                   // No token IDs
  batch.embd = [emb_vec1, emb_vec2, ...];  // Direct embedding vectors

  // Model skips embedding lookup entirely
```

So I understand the concept but it was not clear to me when we would use direct embeddings.
One use case is in mulit-modal models where we might have images encoded as embeddings. For
example there might have been an image encoder that project the image into the embedding
vector space of the text model, in which case we don't have token ids but instead we aldready
have the embedding vectors for the image.


So this is a field similar to the `llama_batch.token` field which is a pointer
to the input tokens. So we might have a prompt which we tokenize which is bacically
splitting the prompt into tokens and looking them up in the models
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

Now, with that in mind when we have `batch.embd` set to point to embeddings
instead of having tokens in the batch, the `inp_embd` tensor will be created
directly as a 2d tensor of float32 values:
```c++
        lctx.inp_embd = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, batch.n_tokens);
        inpL = lctx.inp_embd;
        ggml_set_input(lctx.inp_embd);
```
And later in the `llama_set_inputs` function the `inp_embd` tensor will be populated
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
My initial thought about the use case for using this field was that we could take the
result of a program like `llama_embedding` which can output context aware embeddings and
then use these embeddings, perhaps at some later point, to perform inference based on these
embeddings, similar to how we would used a token prompt. 

In this case what I am trying to use context aware embeddings, so they had already
been processed by the llama.cpp. But as we saw above these would then be used
as inputs to the model and go through all the layers of the model (with the
self-attention and feed-forward layers) and then we would have predict the
next token based on those embeddings. I was somewhat concerned about this as these
embeddings will go through the model layers and self-attention and feed-forward
layers will operate on these embeddings. 

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

Then in the example [embeddings.cpp](../fundamentals/llama.cpp/src/embeddings.cpp)
I did the same thing and extracted the embeddings:
```
embedding 0   0.400597  -0.190815   0.222652   0.074640  -0.191031 
embedding 1   1.491324   0.126824   1.260321  -0.551602   0.151728 
embedding 2   0.130336  -0.622851   2.395198   1.474568  -1.747949 
embedding 3  -0.708625  -0.107112  -0.749924   0.901563  -1.643292 
embedding 4  -1.278546   0.123264  -1.812276  -0.099625  -3.626511 
embedding 5   1.063118  -2.252861  -2.212228   0.937880  -4.282171
```
So that looks pretty good and these are stored in `token_embeddings`.

Now, I was not sure if I should use the last token embedding or all of them for
inference (not that like I mentioned above this might be completely wrong and
not supposed to work like this at all but I'm including this to clarify what I'm
doing). Using all of the embeddings I then created a new context for inference
and the first batch contains the above embeddings.
The I try to peform inference on this and the sampled token is:
```console
token_seq: 4309 : token_str [ Lo]
```
And this looks promising as it is the token `Lo` from the prompt `What is LoRA?`
. The I use this token with a new decode to try to continue the inference I 
start to get "garbage" tokens:
```console
token_seq: 7228 : token [PA]
Inference: token: 7228, pos: 7 
token_seq: 7228 : token [PA]
Inference: token: 7228, pos: 8 
token_seq: 7228 : token [PA]
Inference: token: 7228, pos: 9 
token_seq: 7228 : token [PA]
Inference: token: 7228, pos: 10 
token_seq: 7228 : token [PA]

Generated output:
 LoPAPAPAPAPA
```

Inspecting the logits for this second inference (when the token id is 4309) I
see this:
```console
(gdb) p cur[7228]
$54 = {id = 7228, logit = 14.2697344, p = 0}
(gdb) p ctx.model.vocab.id_to_token[7228]
$55 = {text = "PA", score = -6969, attr = LLAMA_TOKEN_ATTR_NORMAL}
```

So after to debugging I realized that I was using a chat/instruct model and in this
case llama 2 which requires a template matching what the model was trained on. 
```c++
    std::string model_path = "models/llama-2-7b-chat.Q4_K_M.gguf";
    std::string prompt = "<s>[INST] <<SYS>>\n\n<</SYS>>\n\nWhat is LoRA? [/INST]";
```
This took care of the above "PA" token problem but the output is still not great but
the predictions are closer to the context of the original prompt "What is LoRA?":
```console
Top 5 logits:
Token 4309 ( Lo): 13.126865
Token 3410 (Lo): 11.235941
Token 322 ( and): 10.326930
Token 13 (
): 9.867510
Token 297 ( in): 9.695880
token_seq: 4309 : token_str [ Lo]
Inference: token: 4309, pos: 6

Top 5 logits:
Token 29949 (O): 13.566551
Token 29877 (o): 12.267788
Token 29892 (,): 11.640937
Token 3410 (Lo): 11.488244
Token 13 (
): 10.990350
token_seq: 29949 : token [O]
Inference: token: 29949, pos: 7
...
```



_wip_

