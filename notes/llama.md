## Llama
LLaMA (Large Language Model Meta AI) is a large language model from Meta. So
this is a model which means that it contains a binary file with the weights and
biases of the model. These models come in different sizes and are trained on
 different datasets. The larger the model the more data it has been trained on
and the more accurate it is.

### Llama 2
Is really a family of pre-trained models in various scales (the number of
weights). From 7B to 70B.

It is based on the transformer architecture which some improvements like:
* RMSNorm pre-normalization (apperently used by GPT-3)
* SwiGLU activation function (apperently from Google's PaML)
* multi-query attention instead of multi-head attention
* Rotary Positional Embeddings (RoPE) instead of standard positional embeddings
  (apperently inspired by GPT-Neo), see [rope.md](./rope.md)
* AdamW optimizer 

TODO: I'm not familiar with any of the above so this so look into these
separately.

I've seen the achitecture of a transformer where there is an encoder and a
decoder. But my understanding of Llama is that there is only an encoder.
```
                     +-----------+
                     | Softmax   |
                     +-----------+
                          ↑
                     +-----------+
                     | Linear    |
                     +-----------+
                          ↑
                     +-----------+
                     | RMS Norm  |
                     +-----------+
                          ↑
                          |
                          |
  +-----------------------|--------------------------------------------+
  |      +--------------->+                                            |
  |      |                ↑                                            |
  |      |       +--------------------+                                |
  |      |       | Feed Forward SwiGLU|                                |
  |      |       +--------------------+                                |
  |      |                ↑                                            |
  |      |       +--------------------+                                |
  |      |       | RMS Norm           |                                |
  |      |       +--------------------+                                |
  |      |                ↑                                            |
  |      |                |                                            |
  |      +----------------|                                            |
  |    +----------------->+                                            |
  |    |                  ↑                                            |
  |    |   +-----------------------------------------------+           | N times
  |    |   | Self Attention (Grouped Multi Query Attention)|           |
  |    |   | with KV Cache                                 |           |
  |    |   +-----------------------------------------------+           |
  |    |     ↑ ROPE            ↑ ROPE                     ↑            |
  |    |    Q|                K|                         V|            |
  |    |     |                 |                          |            |
  |    |     -----------------------------------------------           |
  |    |                       |                                       |
  |    |             +--------------------+                            |
  |    |             | RMS Norm           |                            |
  |    |             +--------------------+                            |
  |    |                       |                                       |
  |    +-----------------------|                                       |
  |                            |                                       |
  +----------------------------|---------------------------------------+
                               |
                     +--------------------+
                     | Embeddings         |
                     +--------------------+
```

### llama.cpp
[llama.cpp](https://github.com/ggerganov/llama.cpp) is a library written in c
and contains useful programs/examples that can be used to run inference on a
Llama model, as well as quantize a model.
So it can be used to run inference on a model, or to quantize a model and there
are examples in the examples directory for how to do this.

#### Sources
This is just for getting an orientation of the headers and source cpp files
that are used. 

It can be run locally:
```console
$ cd ai/llama
$ make
```
Next we need to download a model to use and store it in the models directory.
I tried the following model:
```
https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q4_0.gguf
```

This the Llama 2 model trained on 13B tokens of chat data. It is a GGUF format
which is suitable for CPU usage. More details about GGUF can be found in
[gptq.md](gptq.md).

Example of running:
```console
$ cd ~/ai/llama.cpp
$ ./main -m models/llama-2-13b-chat.Q4_0.gguf --prompt "What is LoRA?</s>"
```
`main` can be found in examples/main/main.cpp and the output of the log
statements can be found in main-timestamp.log.

We can build the main executable with debug symbols enabled:
```console
$ env GGML_DEBUG=1 LLAMA_DEBUG=1 DEBUG=1 make -B main
```
After that we are able to run the main executable using a debugger:
```console
$ gdb --args ./main -m models/llama-2-13b-chat.Q4_0.gguf --prompt "What is LoRA?</s>"
Reading symbols from ./main...
(gdb) 
```
Now we can break in main and then run the program:
```console
(gdb) break main
Breakpoint 1 at 0x40a4c9: file examples/main/main.cpp, line 105.
(gdb) run
```
The first line of code we encounter is:
```c
  gpt_params params;
```
We can inspect the type of this variable:
```console
(gdb) ptype gpt_params
type = struct gpt_params {
    uint32_t seed; // Random seed value
    int32_t n_threads; 
    int32_t n_predict;
    int32_t n_ctx;
    int32_t n_batch;
    int32_t n_keep;
    int32_t n_draft;
    int32_t n_chunks;
    int32_t n_gpu_layers;
    int32_t main_gpu;
    float tensor_split[1];
```
I don't intend to step through the code but merely have the ability to inspect/
debug it at a later time.

Sampling parameters.

```console
(gdb) p *ctx
$6 = {mem_size = 16, mem_buffer = 0x65c360, mem_buffer_owned = true, no_alloc = false, no_alloc_save = false,
  n_objects = 0, objects_begin = 0x0, objects_end = 0x0, scratch = {offs = 0, size = 0, data = 0x0},
  scratch_save = {offs = 0, size = 0, data = 0x0}}
(gdb) ptype *ctx
type = struct ggml_context {
    size_t mem_size;
    void *mem_buffer;
    _Bool mem_buffer_owned;
    _Bool no_alloc;
    _Bool no_alloc_save;
    int n_objects;
    ggml_object *objects_begin;
    ggml_object *objects_end;
    ggml_scratch scratch;
    ggml_scratch scratch_save;
}
```
So this struct contains pointers to memory buffers which is where all tensor
will be allocated.


#### Presence Penalty
The presence penalty is a hyperparameter that is used control the absence or
presence of new tokens in the generated text.
A value of 0 has no effect on token generation. A negative value will encourage
the model to generate new tokens. A positive value will encourage the model to
not generate new tokens.

#### llm-chain-llama
llm-chain have what they refer to as drivers, at the time of this writing there
are two drivers: OpenAI and Llama. Llama uses a binding to llama.cpp, and is
is created using bindget. The crate llm-chain-llama-sys contains the binding
and llm-chain-llama contains Rust API.

#### llama_batch
This struct holdes `input` data for llama_decode and is defined as:
```c++
    typedef struct llama_batch {
        int32_t n_tokens;

        llama_token  * token;
        float        * embd;
        llama_pos    * pos;
        llama_seq_id * seq_id;
        int8_t       * logits;
    } llama_batch;
```
The `n_tokens` is a counter of the number of tokens that this batch_contains.

A llmm_batch is simlilar to the contept of context we talked about
[llm.md](../../notes/llm.md#context_size). Below we are adding the input query
tokens to this batch/context. So it will initially just contain the tokens for
our query. But after running the inference, we will append the next token to the
batch and run the inference again and then run the inferencex again to predict
the next token, now with more context (the previous token).

The `embd` is the embedding of the tokens (I think). So this was not obvious to
me at first, but recall that the tokens just integer representations of
works/subwords, like a mapping. But they don't contains any semantic
information. Recall that this is data which is setup as input to for
llama_decode and I think this is used when embeddings are already available
perhaps. TODO: verify this.
The `pos` is the position of the tokens in the sequence.


### llama_batch
This struct holdes `input` data for llama_decode. For example, if we pass in
a prompt of "What is LoRA" that would first be tokenized and then the tokens
will be added to the batch. An example of this can be found in
[simple-prompt.cpp](../fundamentals/llama.cpp/src/simple-prompt.cpp).

An instance of a batch contains a count of the number of tokens (or embeddings)
that this batch holds. In the above case n_tokens would be 7.
```c++
    llama_batch batch = llama_batch_init(512, /*embd*/ 0, /*n_seq_max*/ 1);
    for (size_t i = 0; i < input_tokens.size(); i++) {
        // the token of this batch entry.
        batch.token[batch.n_tokens] = input_tokens[i];
        // the position in the sequence of this batch entry.
        batch.pos[batch.n_tokens] = i,
        // the sequence id (if any) of this batch entry.
        batch.n_seq_id[batch.n_tokens] = seq_ids.size();
        for (size_t s = 0; s < seq_ids.size(); ++s) {
            batch.seq_id[batch.n_tokens][s] = seq_ids[s];
        }
        // Determins if the logits for this token should be generated or not.
        batch.logits[batch.n_tokens] = false;
        // Increment the number of tokens in the batch.
        batch.n_tokens++;
    }
```
We can take a look at the third token in this batch using:

```console
$ gdb --args ./simple-prompt
(gdb) br llama.cpp:5514
(gdb) r
(gdb) p batch
$10 = {n_tokens = 7, token = 0xc4fb20, embd = 0x0, pos = 0x894ab0, n_seq_id = 0x897d10, seq_id = 0x898520, 
  logits = 0x895b50 "", all_pos_0 = 0, all_pos_1 = 0, all_seq_id = 0}

(gdb) p batch.token[3]
$6 = 4309

(gdb) p batch.pos[3]
$7 = 3
```

Now, I think I understand that position is the position of this token in the
input sequence.

But I'm not sure what `n_seq_id` is. There is one for each token in the batch
so it has a size of 7 (n_tokens). 
```console
(gdb) p batch.n_seq_id[3]
$13 = 1

(gdb) p *batch.seq_id[3]
$9 = 0
```
Lets see if we can figure this out by looking at `llama_decode` and how it
uses these sequence values. If we set these pointers to null and rerun we
will be able to see how llama_decode uses these values:
```c++
    batch.n_seq_id = nullptr;
    batch.seq_id = nullptr;
```
We have the following if statement in llama_decode:
```c++
    std::vector<int32_t> n_seq_id;
    std::vector<llama_seq_id *> seq_id_arr;
    std::vector<std::vector<llama_seq_id>> seq_id;

    if (batch.seq_id == nullptr) {
        n_seq_id.resize(n_tokens);
        seq_id.resize(n_tokens);
        seq_id_arr.resize(n_tokens);
        for (uint32_t i = 0; i < n_tokens; i++) {
            n_seq_id[i] = 1;
            seq_id[i].resize(1);

            seq_id[i][0] = batch.all_seq_id;
            seq_id_arr[i] = seq_id[i].data();
        }

        batch.n_seq_id = n_seq_id.data();
        batch.seq_id = seq_id_arr.data();
    }
```
First notice that 3 vectors are initialized and these will be populated if
the batch.seq_id is null. Each entry in the `n_seq_id` vector will be set to 1,
and the `seq_id` will be resized to that size (1) as well. Next, the actual
value of the sequence id is set to `batch.all_seq_id` which is 0.
Finally `batch.n_seq_id` and `batch.seq_id` are set to point to this data.
Hmm, so the batch position tells use the position of the token in this batch.

```
batch tokenized from "What is LoRA", n_tokens = 7

batch.token[0] = 0      batch.pos[0] = 0  n_seq_id[0] = 1 seq_id[0][0] = 0
batch.token[1] = 1      batch.pos[1] = 1  n_seq_id[1] = 1 seq_id[1][0] = 0
batch.token[2] = 338    batch.pos[2] = 2  n_seq_id[2] = 1 seq_id[2][0] = 0
batch.token[3] = 4309   batch.pos[3] = 3  n_seq_id[3] = 1 seq_id[3][0] = 0
batch.token[4] = 4717   batch.pos[4] = 3  n_seq_id[4] = 1 seq_id[4][0] = 0
batch.token[5] = 29973  batch.pos[5] = 5  n_seq_id[5] = 1 seq_id[5][0] = 0
batch.token[6] = 13     batch.pos[6] = 6  n_seq_id[6] = 1 seq_id[6][0] = 0
                                                          (size=1)       ↑
                                                                    sequence number
```
It is possible to set the values of the sequences to something else. For
example, if we set the sequence id of the first token to 1, and the sequence id
of the second token to 2, then we will have two sequences in this batch.
```c++
    batch.n_seq_id[0] = 1;
    batch.seq_id[0][0] = 1;
    batch.n_seq_id[1] = 1;
    batch.seq_id[1][0] = 2;
```
I'm still not sure how this is useful.
