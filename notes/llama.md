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
$ cd ai/llama.cpp
$ make -j8 
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


#### Repeat Penalty
The repeat penalty is a hyperparameter that is used to control the repetition
of tokens in the generated text. Setting this to 1 will not have any effect on
token generation. A value of 0 will encourage the model to repeat tokens. A
value greater than 1 will encourage the model to not repeat tokens.

I ran into this issue with the llam-chains-chat-demo where the llm would just
repeat the new-line token over and over until the context size was reached.
Adding a repeat penalty of 1.1 fixed this issue and is something to be aware of.

#### Frequency Penalty
The frequency penalty is a hyperparameter that is used to control the frequency
of tokens in the generated text. Setting this to 1 will not have any effect on
token generation. A value of 0 will encourage the model to generate tokens that
are more frequent. A value greater than 1 will encourage the model to generate
tokens that are less frequent.


#### llm-chain-llama
llm-chain have what they refer to as drivers, at the time of this writing there
are two drivers: OpenAI and Llama. Llama uses a binding to llama.cpp, and is
is created using bindget. The crate llm-chain-llama-sys contains the binding
and llm-chain-llama contains Rust API.

#### llama_batch
This struct holdes `input` data for `llama_decode` and is defined as:
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
The `n_tokens` is a counter of the number of tokens that this batch contains.

A `llama_batch` is simlilar to the contept of context we talked about
[llm.md](../../notes/llm.md#context_size). Below we are adding the input query
tokens to this batch/context. So it will initially just contain the tokens for
our query. But after running the inference, we will append the next token to the
batch and run the inference again and then run the inference again to predict
the next token, now with more context (the previous token).

The `embd` is the embedding of the tokens (I think). So this was not obvious to
me at first, but recall that the tokens just integer representations of
works/subwords, like a mapping. But they don't contains any semantic
information. Recall that this is data which is setup as input for llama_decode
and I think this is used when embeddings are already available perhaps.
TODO: verify this.
The `pos` is the position of the tokens in the sequence.

### Key-Value Cache
This section tries to explain the key-value cache used in the llama 2
architecture. This is the caching of the key and value matrices that are used
in the attention architecture.

First lets take a look at inference without the key-value cache. Now in the
following example we are starting with input tokens with a dimension of 2.
So for each token that we have, we have a vector of size 2. And we will assume
that the first token is the start of sentence token and that the model is going
to predict the next token. What it predicts does not matter to make the point
here so just ignore the actual values.
```
Attention(Q, K V) = softmax(QK^T / sqrt(d_k)) V

input token [1 2] (Start of Sentence)

Time=1

[  Q       K^T      QK^T  ] *  V  
 [1 2]   . [1]    = [5]     * [1 2] = [5 10]
           [2]    

  (dot product)   (scalar multiplication because [5] is a single number)
```
So we can see that we have performed the dot product of the query and the key
and then multiplied that by the value matrix. Lets pretend that the output of
this is then the probability of the next token, and that token was selected from
the models vocabulary (we are ignoring softmax etc here).

So for the next token we then append the predicted token to the input tokens
and run the inference again:
```
Time=2

[  Q       K^T      QK^T  ]  *   V  
 [1 2 ]  . [1  5] = [5  25 ] * [1  2] = [5*1 + 25*5     5*2 +  25*10] = [130  260]
 [5 10]    [2 10]   [25 125]   [5 10]   [25*1 + 125*5  25*2 + 125*10]   [650 1300]
  (dot product)     (matrix multiplication)
```
Once again we will take the last token and append it to the input tokens and
run the inference again:
```
Time=3

[  Q                 K^T      QK^T  ]  *   V  
 [1     2 ]   . [1  5  650] = [5    25    3250   ] =
 [5     10]     [2 10 1300]   [25   125   16250  ]
 [650 1300]                   [3250 16250 2210000]
```
Notice that we have calculated the dot product for the first and second token
again!

The transformer architecture needs to have all the previous tokens in the
sequence to predict the next token but we don't have to recalculate the dot
product every time. We can cache the key and value matrices and then just
use a single input token, the last token predicted and not append that token to
the input:
```
Time=2
[  Q       K^T      QK^T  ]  *   V  
 [5 10 ]  . [1  5] = [25  125] * [1 5 ] = [650 1300]
            [2 10]               [2 10]
```
By adding the output token to the `Key` and the `Value` matrices we perform
fewer operations. This is the key-value cache.
So for every token processed it needs to be added to the Key and Value cache
matrices to be use in the next predition.

Let's take a look at how this is implemented in llama.cpp. I'll be using
simple-prompt to demonstrate this.
```console
$ gdb --args ./simple-prompt
(gdb) br simple-prompt.cpp:34
(gdb) r
(gdb) f
#1  0x0000000000408fab in main (argc=1, argv=0x7fffffffd198) at src/simple-prompt.cpp:34
34	    llama_context * ctx = llama_new_context_with_model(model, ctx_params);
(gdb) s
gdb) l
8736	        return nullptr;
8737	    }
8738	
8739	    llama_context * ctx = new llama_context(*model);
(gdb) p ctx.kv_self
$5 = {has_shift = false, head = 0, size = 0, used = 0, n = 0, cells = std::vector of length 0, capacity 0, k = 0x0, 
  v = 0x0, ctx = 0x0, buf = {data = 0x0, size = 0, fallback = false}}
```
At this stage the kv_self is uninitialized. We can inspect this struct using:
```console
gdb) ptype ctx.kv_self
type = struct llama_kv_cache {
    bool has_shift;
    bool do_defrag;
    bool do_copy;
    bool recurrent;
    uint32_t head;
    uint32_t size;
    uint32_t used;
    uint32_t n;
    ggml_type type_k;
    ggml_type type_v;
    std::vector<llama_kv_cell> cells;
    std::vector<ggml_tensor*> k_l;
    std::vector<ggml_tensor*> v_l;
    std::vector<ggml_context*> ctxs;
    std::vector<ggml_backend_buffer*> bufs;
  public:
    size_t total_size(void) const;
    ~llama_kv_cache(void);
}
```
We can see we have two vectors of  pointers to ggml_tensor. These are the key
and value matrices for each layer.

A little further down we have:
```console
(gdb) s
8789 if (!llama_kv_cache_init(ctx->model.hparams, ctx->kv_self, memory_type, cparams.n_ctx, model->n_gpu_layers)) {
Notice that we are passing in `ctx->kv_self`, and the cparams.n_ctx which is
the context length set to 1024 in this case.
```console
(gdb) s
1518	static bool llama_kv_cache_init(
(gdb) l
1519	        const struct llama_hparams & hparams,
1520	             struct llama_kv_cache & cache,
1521	                         ggml_type   wtype,
1522	                          uint32_t   n_ctx,
1523	                               int   n_gpu_layers) {
1524	    const uint32_t n_embd  = hparams.n_embd_gqa();
1525	    const uint32_t n_layer = hparams.n_layer;
1526	
1527	    const int64_t n_mem      = n_layer*n_ctx;
1528	    const int64_t n_elements = n_embd*n_mem;
```
The first line had be asking what `gqa` is and I think this stands for
grouped query attention. 
```console
(gdb) s
1532	    cache.head = 0;
(gdb) s
1533	    cache.size = n_ctx;
(gdb) s
1534	    cache.used = 0;
(gdb) s
1536	    cache.cells.clear();
(gdb) p cache
$20 = (llama_kv_cache &) @0xc624c8: {has_shift = false, head = 0, size = 1024, used = 0, n = 0, 
  cells = std::vector of length 0, capacity 0, k = 0x0, v = 0x0, ctx = 0x0, buf = {data = 0x0, size = 0, 
    fallback = false}}

1536	    cache.cells.clear();
(gdb) n
1537	    cache.cells.resize(n_ctx);
(gdb) n
(gdb) p cache
$21 = (llama_kv_cache &) @0xc624c8: {has_shift = false, head = 0, size = 1024, used = 0, n = 0, 
  cells = std::vector of length 1024, capacity 1024 = {{pos = -1, delta = 0, seq_id = std::set with 0 elements}, {
      pos = -1, delta = 0, seq_id = std::set with 0 elements}, {pos = -1, delta = 0, 
      seq_id = std::set with 0 elements}, {pos = -1, delta = 0, seq_id = std::set with 0 elements}, {pos = -1, 
      ...
1544	    params.mem_buffer = cache.buf.data;
(gdb) s
1545	    params.no_alloc   = false;
(gdb) s
1547	    cache.ctx = ggml_init(params);
```
So we can see here that we are going to initialize context for ggml. I did not
notice that `ggml_context *ctx` was a member of `llama_kv_cache`.
Next we are going to create a one dimensional tensor of GGML_TYPE_F16 (half
precision float) with 209715200 elements.
```console
1554	    cache.k = ggml_new_tensor_1d(cache.ctx, wtype, n_elements);
(gdb) p n_elements
$31 = 209715200
(gdb) p wtype
$32 = GGML_TYPE_F16
```
Hmm, the size of the tensor don't make sense to me yet. The 1d tensor is like
a list of number and it's size is 209715200. And the type of these slots is
F16 so that would be 2 bytes per slot, so 16 bytes per slot.
```console
llama_new_context_with_model: kv self size  =  800.00 MiB
```

```console
gdb) s
8804	            ctx->logits.reserve(hparams.n_vocab);
(gdb) p hparams.n_vocab
$40 = 32000
```
That is pretty much it for the intialization of the llama_context. This will
return us to simple_prompt.cpp:
```console
4	    llama_context * ctx = llama_new_context_with_model(model, ctx_params);
35	    if (ctx == NULL) {
36	        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
37	        return 1;
38	    }
```
Now, lets look what happens when decode is called and how this interacts with
the key-value cache.
```console
(gdb) br simple-prompt.cpp:115
Breakpoint 2 at 0x40949e: file src/simple-prompt.cpp, line 115.
(gdb) c
Continuing.
batch.n_tokens: 6
batch.tokens: [1, 1724, 338, 4309, 4717, 29973, ]
prompt: What is LoRA?
Breakpoint 2, main (argc=1, argv=0x7fffffffd198) at src/simple-prompt.cpp:115
115	    if (llama_decode(ctx, batch) != 0) {
```
And the batch looks like this:
```console
(gdb) p batch
$54 = {n_tokens = 6, token = 0xc63980, embd = 0x0, pos = 0x8a86f0, n_seq_id = 0x8ab7b0, seq_id = 0x8abfc0, 
  logits = 0x8a9790 "", all_pos_0 = 0, all_pos_1 = 0, all_seq_id = 0}
(gdb) s
5678	    const auto n_batch = cparams.n_batch;
(gdb) s
(gdb) p n_batch
$56 = 512
```
So, n_batch is the maximum number of tokens that can be in a single batch, and
n_tokens is the number of tokens in the current batch.
```console
(gdb) n
5682	    int n_threads = n_tokens == 1 ? cparams.n_threads : cparams.n_threads_batch;
```
I found this a little interesting and because I've always called decode with
a number of tokens, never a single token. But thinking back the example with
the key-value cache and how it would pass in a single token as the input but
the key and value matrices would contain all the previous tokens.
```console
(gdb) n
5695	    auto & kv_self = lctx.kv_self;

5734	    // if we have enough unused cells before the current head ->
5735	    //   better to start searching from the beginning of the cache, hoping to fill it
5736	    if (kv_self.head > kv_self.used + 2*n_tokens) {
5737	        kv_self.head = 0;
5738	    }
5739	
5740	    if (!llama_kv_cache_find_slot(kv_self, batch)) {
```
```console
1584	// find an empty slot of size "n_tokens" in the cache
1585	// updates the cache head
1586	// Note: On success, it's important that cache.head points
1587	// to the first cell of the slot.
1588	static bool llama_kv_cache_find_slot(
1589	           struct llama_kv_cache & cache,
1590	        const struct llama_batch & batch) {
1591	    const uint32_t n_ctx    = cache.size;
1592	    const uint32_t n_tokens = batch.n_tokens;
1593	
1594	    if (n_tokens > n_ctx) {
1595	        LLAMA_LOG_ERROR("%s: n_tokens=%d > n_ctx=%d\n", __func__, n_tokens, n_ctx);
1596	        return false;
1597	    }
```
In this case we have the max context size in token of 1024 and the number of
tokens in the batch is 6:
```console
(gdb) p n_ctx
$69 = 1024
(gdb) p n_tokens
$70 = 6
```
The number of tokens in the batch cannot exceed the max context size.
```console
(gdb) l
1599	    uint32_t n_tested = 0;
1600	
1601	    while (true) {
1602	        if (cache.head + n_tokens > n_ctx) {
1603	            n_tested += n_ctx - cache.head;
1604	            cache.head = 0;
1605	            continue;
1606	        }
```
So we are going to loop and the first thing we to is check if the head plus the
number of tokens in the batch exceed the max number of tokens allowed.  If this
is the case then n_tested is incremented with the max context size minus the
cache head.
Lets pretent that we have a head that is 1020 and the number of tokens is 6 and
n_ctx is 1024. Then 1020+6=1026 and 1026 > 1024. And n_tested will become
1024-1020=4. And the head will be set to 0. And then the loop will continue but
this time head will be zero. And the if statement will compare 6 > 1024 which
is false and skip the body of the if statement.
```console
1608	        bool found = true;
1609	        for (uint32_t i = 0; i < n_tokens; i++) {
1610	            if (cache.cells[cache.head + i].pos >= 0) {
1611	                found = false;
1612	                cache.head += i + 1;
(gdb) l
1613	                n_tested   += i + 1;
1614	                break;
1615	            }
1616	        }
```
So we are going to loop over all the 6 tokens in the batch.
```console
(gdb) p i
$83 = 0
(gdb) p cache.head
$84 = 0

(gdb) p cache.cells[cache.head + i].pos
$85 = -1
```
cache.cells is a vector of size 1024, the max number of tokens allowed. And
each entry in this  vector is currently not set, its position is -1. So in our
case this will false and the if block will not be executed. So this is making
sure that from the current head there are n_tokens number of slots available.
That will lead us to the following code:
```console
1628	    for (uint32_t i = 0; i < n_tokens; i++) {
1629	        cache.cells[cache.head + i].pos = batch.pos[i];
1630	
1631	        for (int32_t j = 0; j < batch.n_seq_id[i]; j++) {
1632	            cache.cells[cache.head + i].seq_id.insert(batch.seq_id[i][j]);
1633	        }
1634	    }
1635	
1636	    cache.used += n_tokens;
1637	
1638	    return true;
```
And it makes sence that we again will loop over the 6 tokens in the batch and
now add them to the cells. 
```console
gdb) p cache.cells
$106 = std::vector of length 1024, capacity 1024 = {
{pos = 0, delta = 0, seq_id = std::set with 1 element = {[0] = 0}},
{pos = 1, delta = 0, seq_id = std::set with 1 element = {[0] = 0}},
{pos = 2, delta = 0, seq_id = std::set with 1 element = {[0] = 0}},
{pos = 3, delta = 0, seq_id = std::set with 1 element = {[0] = 0}},
{pos = 4, delta = 0, seq_id = std::set with 1 element = {[0] = 0}},
{pos = 5, delta = 0, seq_id = std::set with 1 element = {[0] = 0}},
```
That is it for finding a slot in the key-value cache.
```console
(gdb) s
5747	    kv_self.n = std::min(
                (int32_t) cparams.n_ctx,
                std::max(32, GGML_PAD(llama_kv_cache_cell_max(kv_self), 32))
            );
```
A cell is considered in use if its position is greater than or equal to zero and
it's sequence id is not empty. So that should return 6 in our case:
```console
(gdb) p llama_kv_cache_cell_max(kv_self)
$108 = 6
(gdb) p cparams.n_ctx
$109 = 1024
(gdb) p kv_self.n
$110 = 32
```
What is kv_self.n?  

```console
(gdb) f
#0  llama_decode_internal (lctx=..., batch=...) at llama.cpp:5751
5751	    ggml_allocr_reset(lctx.alloc);
(gdb) n
5753	    ggml_cgraph * gf = llama_build_graph(lctx, batch);
```
So we are building a compute graph for this batch.
```console
(gdb) n
5757	    struct ggml_tensor * res        = gf->nodes[gf->n_nodes - 1];
```
The result tensor is the last node in the graph.
The embedding tensor is the second to last node in the graph:
```console
(gdb) n
5758	    struct ggml_tensor * embeddings = gf->nodes[gf->n_nodes - 2];
```
```console
(gdb) n
5816	    ggml_graph_compute_helper(lctx.work_buffer, gf, n_threads);
```
This will now run the compute graph which involves starting threads:
```console
gdb) n
[New Thread 0x7ffe072eb6c0 (LWP 2085673)]
[New Thread 0x7ffe06aea6c0 (LWP 2085674)]
[New Thread 0x7ffe062e96c0 (LWP 2085675)]
[Thread 0x7ffe062e96c0 (LWP 2085675) exited]
[Thread 0x7ffe06aea6c0 (LWP 2085674) exited]
[Thread 0x7ffe072eb6c0 (LWP 2085673) exited]
5825	        if (kv_self.has_shift) {
```
Next, we have:
```console
5823	    // update the kv ring buffer
5824	    {
5825	        if (kv_self.has_shift) {
5826	            kv_self.has_shift = false;
5827	            for (uint32_t i = 0; i < kv_self.size; ++i) {
5828	                kv_self.cells[i].delta = 0;
5829	            }
```
The `llama_kv_cache` is a ringbuffer and when the buffer is full and we need to
add data the oldest data is overwritten which I believe is called shifting.
This false in our case.
```console
5832	        kv_self.head += n_tokens;
(gdb) s
5835	        if (kv_self.head >= kv_self.size) {
(gdb) p kv_self.head
$118 = 6
(gdb) p kv_self.size
$119 = 1024
```
So that was the initial prompt which has now been decode and then we will use
the logits to predict the next token. This will be a single token which we will
then pass into llama_decode:
```console
(gdb) 
206	        if (llama_decode(ctx, batch)) {
(gdb) p batch
$125 = {n_tokens = 1, token = 0xc63980, embd = 0x0, pos = 0x8a86f0, n_seq_id = 0x8ab7b0, seq_id = 0x8abfc0, 
  logits = 0x8a9790 "\001", all_pos_0 = 0, all_pos_1 = 0, all_seq_id = 0}
(gdb) p batch.pos[0]
$126 = 6
```
Notice that this time around the kv_self is:
```console
(gdb) p lctx.kv_self
$128 = {has_shift = false, head = 6, size = 1024, used = 6, n = 32,
...
```
I'm trying to understand where the kv_self is used. I can see that it is updated
as we saw above. Where is the Q and the K^T calculation done?
For this we have to take a look at how the compute graph is built. This is done
in build_llama.
_wip_

### llama_batch
This struct holdes `input` data for `llama_decode` and is defined as:
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
The `n_tokens` is a counter of the number of tokens that this batch contains.

A `llama_batch` is simlilar to the contept of context we talked about
[llm.md](../../notes/llm.md#context_size). Below we are adding the input query
tokens to this batch/context. So it will initially just contain the tokens for
our query. But after running the inference, we will append the next token to the
batch and run the inference again and then run the inference again to predict
the next token, now with more context (the previous token).

The `embd` is the embedding of the tokens (I think). So this was not obvious to
me at first, but recall that the tokens are just integer representations of
works/subwords, like a mapping of the token to an index in the vocabulary. But
they don't contains any semantic information. Notice that embed is a pointer
to float and not an integer like the token field.
TODO: clarify the embd field.

The `pos` is the position of the tokens in the sequence.

This struct holds `input` data for llama_decode. For example, if we pass in
a prompt of "What is LoRA" that would first be tokenized and then the tokens
will be added to the batch. An example of this can be found in
[simple-prompt.cpp](../fundamentals/llama.cpp/src/simple-prompt.cpp).

An instance of a batch contains a count of the number of tokens (or embeddings)
that this batch holds. In the above case `n_tokens` would be 7.
```c++
    llama_batch batch = llama_batch_init(512, /*embd*/ 0, /*n_seq_max*/ 1);
    for (int i = 0; i < n_tokens; i++) {
        // the token of this batch entry.
        batch.token[i] = input_tokens[i];
        // the position in the sequence of this batch entry.
        batch.pos[i] = i,
        // the number of sequence id's of this batch entry.
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;  // the sequence id
        // Determins if the logits for this token should be generated or not.
        batch.logits[i] = false;
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

### tensor-split
There is model_param value which is a pointer to floats and the size of this
array is the value of LLAMA_MAX_DEVICES. This value is defined in llama.h:
```c++

#ifdef GGML_USE_CUBLAS                                                             
#include "ggml-cuda.h"                                                          
#define LLAMA_MAX_DEVICES GGML_CUDA_MAX_DEVICES                                    
#else                                                                              
#define LLAMA_MAX_DEVICES 1                                                        
#endif // GGML_USE_CUBLAS     
...

struct llama_model_params {                                                 
    int32_t n_gpu_layers; // number of layers to store in VRAM                 
    int32_t main_gpu;     // the GPU that is used for scratch and small tensors
    const float * tensor_split; // how to split layers across multiple GPUs (size: LLAMA_MAX_DEVICES)
    ...
}
```
So in the case where cuBLAS (CUDA Basic Linear Algebra Subroutine) is used the
size of this array will be the maximum number of devices that can be used.
The values in this array will be of type float and and would be how the layers
of the neural network should be split accorss the devices. This allows for
specifying that more layers should be stored on one device than another. For
example [0.7, 0.3] would mean that 70% of the layers should be stored on the
first device and 30% on the second device.


The llama_batch struct looks like this:
```c++
   // Input data for llama_decode
    // A llama_batch object can contain input about one or many sequences
    // The provided arrays (i.e. token, embd, pos, etc.) must have size of n_tokens
    //
    // - token  : the token ids of the input (used when embd is NULL)
    // - embd   : token embeddings (i.e. float vector of size n_embd) (used when token is NULL)
    // - pos    : the positions of the respective token in the sequence
    // - seq_id : the sequence to which the respective token belongs
    // - logits : if zero, the logits for the respective token will not be output
    //
    typedef struct llama_batch {
        int32_t n_tokens;
        llama_token  *  token;
        float        *  embd;
        llama_pos    *  pos;
        int32_t      *  n_seq_id;
        llama_seq_id ** seq_id;
        int8_t       *  logits;

        // NOTE: helpers for smooth API transition - can be deprecated in the future
        //       for future-proof code, use the above fields instead and ignore everything below
        //
        // pos[i] = all_pos_0 + i*all_pos_1
        //
        llama_pos    all_pos_0;  // used if pos == NULL
        llama_pos    all_pos_1;  // used if pos == NULL
        llama_seq_id all_seq_id; // used if seq_id == NULL
    } llama_batch;
```
`n_tokens` is the number of tokens in this batch.
`token` is a int pointer to tokens.
`embd` is a float pointer to embeddings.
`pos` is a pointer to the position of the tokens in the sequence. So each token
in the batch will have a value in this array which is of size `n_tokens`.
`n_seq_id` is the number of sequence ids. I still don't understand what the
sequence ids are used for. Perhaps that are used for parallel processing?
`seq_id` is the sequence id for each token in the batch. So each token will
have a sequence id.

`all_pos_0` is only used if pos is NULL.
`all_pos_1` is only used if pos is NULL.
So if the pos array/pointer is null then the decode function will check for
this condition and populate a std::vector(llama_pos> with the size of n_tokens.
It will then iterate through the range of 0..n_tokens and set the pos values:
```
all_pos_0 = 0
all_pos_1 = 1

   pos[i] = all_pos_0 + i * all_pos_1

   pos[0] = 0 + 0 * 1 = 0
   pos[1] = 0 + 1 * 1 = 1
   pos[2] = 0 + 2 * 1 = 2
```
And we can specify that the position starts with a value other than 0:
```
all_pos_0 = 4
all_pos_1 = 1

   pos[0] = 4 + 0 * 1 = 4
   pos[1] = 4 + 1 * 1 = 5
   pos[2] = 4 + 2 * 1 = 6
```
So this is a just way of specifying the position of the tokens in the sequence
and something we could have done manually ourselves.
Also not that this is deprecated in llama.cpp.
```
        // NOTE: helpers for smooth API transition - can be deprecated in the future
        //       for future-proof code, use the above fields instead and ignore everything below
        //
        // pos[i] = all_pos_0 + i*all_pos_1
        //
        llama_pos    all_pos_0;  // used if pos == NULL
        llama_pos    all_pos_1;  // used if pos == NULL
        llama_seq_id all_seq_id; // used if seq_id == NULL
```

`all_seq_id` is only used if seq_id is NULL.
```c++
    std::vector<int32_t>                   n_seq_id;
    std::vector<llama_seq_id *>            seq_id_arr;
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
So we will first create `n_seq_id`, `seq_id`, and `seq_id_arr` vectors of size
3 in this case. And 'batch.all_seq_id' is 0.
```
seq_id[  [0],
         [0], 
         [0],
      ]
```

Lets start with `n_seq_id` which is an array and each token in a batch will have
an entry in this array. The value in this position specifies the number of
sequences that the token is part of (still not sure exactly what this means of
how it is used but hopefully that will clear up).
Lets say that we have a batch of 3 tokens and the second token is part of two
sequences:
```
n_seq_id[1] = 2;
```
The corresponding entry in the `seq_id` vector will point to a vector of size
2 in that case.
```
seq_id[  [0],
         [1, 2], 
         [0],
      ]
```
One usage of the n_seq_id is in `llama_kv_cache_find_slot`:
```c++
    for (uint32_t i = 0; i < n_tokens; i++) {
        cache.cells[cache.head + i].pos = batch.pos[i];

        for (int32_t j = 0; j < batch.n_seq_id[i]; j++) {
            cache.cells[cache.head + i].seq_id.insert(batch.seq_id[i][j]);
        }
    }

    cache.used += n_tokens;
```
So for each token in the batch an entry at `cache.head + ` will be updated with
the current tokens position. And notice that it will loop through the number
of sequences that the current token has, which is the value of `n_seq_id` which
is a member of th llama_kv_cell struct:
```c++
struct llama_kv_cell {
    llama_pos pos   = -1;
    llama_pos delta = 0;

    std::set<llama_seq_id> seq_id;
};
```

Hmm, I think what I've been missing might be that a batch can contain one or
more tokens, and each token has a position related with it.
For example, we could have an initial prompt that we pass as input put to
`llama_decode` which we know takes a batch:
```
   batch_0:
           n_tokens: 3
           token: [102, 23, 2993]
           pos: [0, 1, 2]
           n_seq_id: [1, 1, 1]
           seq_id: [[0], [0], [0]]
```
And then we call one of the sampling functions, like `llama_sample_greedy`, to
get a new token id. We will then send this token id and have the llm infer
the next token. This time we only send a single token as input:
```
   batch_1:
           n_tokens: 1
           token: [1186]
           pos: [3]
           n_seq_id: [1]
           seq_id: [[0]]
```
But notice that we have updated the position of the token in the sequence to be
3 and the sequence id is still 0. So we are still in the same sequence but
we have moved to the next token in the sequence. And we can continue to do this
until we have a new sequence. For example, we could have a new sequence that
starts with the token 0:
```
   batch_2:
           n_tokens: 1
           token: [0]
           pos: [0]
           n_seq_id: [1]
           seq_id: [[1]]
``` 
Now, how about the case where we have multiple sequence ids in a batch. For
example:
```
sequence_0: Dan loves ice cream
            batch.n_tokens: 4
            batch.pos: [0, 1, 2, 3]
            batch.n_seq_id: [1, 1, 1, 1]
            batch.seq_id: [[0], [0], [0], [0]]
```

An entry in the kv cache contains a position and also an vector of sequence
ids like we also saw above:
```c++
struct llama_kv_cell {
    llama_pos pos   = -1;
    llama_pos delta = 0;

    std::set<llama_seq_id> seq_id;
};
When a batch is processes and added into the kv cache, all the tokens in the
batch will be iterated over and the next available slot will be found, head+i,
will set that kv cache entry's pos to the current tokens position. The same
iteration will also add the sequence ids of the current token to the same kv
cache entry.
We can add a second sequence to a batch by using the following:
```c++
llama_batch batch = llama_batch_init(512, 0, 2);
for (int i = 0; i < n_tokens; i++) {
  batch.token[i] = input_tokens[i];
  batch.pos[i] = i;
  batch.n_seq_id[i] = 2;
  batch.seq_id[i][0] = 0;
  batch.seq_id[i][1] = 1;
  batch.logits[i] = false;
  batch.n_tokens++;
}
```
Now, I can see the usefulness of having sequence ids for example if I start
with one query and then ask a completely different question I want to have it
evaluated separate from the first, but I might also want to come back to the
first.

TODO: Figure out the usage of multiple sequence ids in a batch.

### Tensors
```console
llm_load_print_meta: vocab type       = SPM
llm_load_print_meta: n_vocab          = 32000
llm_load_print_meta: n_merges         = 0
llm_load_print_meta: n_ctx_train      = 4096
llama_model_loader: loaded meta data with 19 key-value pairs and 291 tensors
from ./models/llama-2-7b-chat.Q4_0.gguf (version GGUF V2)

llama_model_loader: - tensor    0:  token_embd.weight q4_0 [4096, 32000, 1, 1]
```
This is the embeddings for the model. Recall that the model has a context size
of 4096 and a vocab size of 32000. So for each token in the vocabulary there
is en embedding with a dimension of 4096.

### Inspecting a token
Sometimes you might have a token id and want to know what it represents. This
can be done opening a program in a debugger. For example:
```console
$ gdb -args ./simple-prompt
Reading symbols from ./simple-prompt...
(gdb) br simple-prompt.cpp:21
Breakpoint 1 at 0x408ea2: file src/simple-prompt.cpp, line 21.
(gdb) r
```
After the model has loaded we can then inspect the token embeddings, in this
case I wanted to know what the token id 29871 represents
```console
(gdb) p model.vocab.id_to_token[29871]
$6 = {text = "▁", score = -1e+09, type = LLAMA_TOKEN_TYPE_NORMAL}
```

### Prompting llama2
This page https://gpus.llm-utils.org/llama-2-prompt-template/, and also
https://huggingface.co/blog/llama2#how-to-prompt-llama-2, specifies that
a prompt look as follows:
```
<s>[INST] <<SYS>>
{your_system_message}
<</SYS>>

{user_message_1} [/INST]
```
This is how the model was trained and so this is what it expects.
This might sound obvious but I ran into this issue when trying to create a
prompt that would use retrieval augmented generation (RAG). I was trying to add
some additional examples of interactions for the model as a system message but
I originally specified them something like this:
````
    let sys_prompt = r#"
[INST] <<SYS>>

{{ system_prompt }}

Only respond with the YAML and nothing else.

Here are some previous interactions between the Assistant and a User:

User: What is RHSA-1820:1234?
Assistant:
```yaml
command: VectorStoreTool
input:
  query: "RHSA-1820:1234"
  limit: 4
```

User: Can you show me the details about advisory RHSA-1721:4231?
Assistant:
```yaml
command: VectorStoreTool
input:
  query: "RHSA-1721:4231"
  limit: 4
```

User: Is is RHSA-1721:4231 about an OpenSSL exploit?
Assistant:
```yaml
command: VectorStoreTool
input:
  query: "RHSA-1721:4231"
  limit: 4
```

Your output should only be YAML and include query, and limit fields. Do not output any other text or other information.

<</SYS>>

{{ user_message }} [/INST]"#;
````
Sometimes the llm would get this right but most of the times it would not and
not create a valid YAML. After a while I relized my mistake and changed the user
messages in the examples to include the `[INST]` and `[/INST]` tags:
````
    let sys_prompt = r#"
[INST] <<SYS>>

{{ system_prompt }}

Only respond with the YAML and nothing else.

Here are some previous interactions between the Assistant and a User:

[INST] User: What is RHSA-1820:1234? [/INST]
Assistant:
```yaml
command: VectorStoreTool
input:
  query: "RHSA-1820:1234"
  limit: 4
```

[INST] User: Can you show me the details about advisory RHSA-1721:4231? [/INST]
Assistant:
```yaml
command: VectorStoreTool
input:
  query: "RHSA-1721:4231"
  limit: 4
```

[INST] User: Is is RHSA-1721:4231 about an OpenSSL exploit? [/INST]
Assistant:
```yaml
command: VectorStoreTool
input:
  query: "RHSA-1721:4231"
  limit: 4
```

Your output should only be YAML and include query, and limit fields. Do not output any other text or other information.

<</SYS>>

{{ user_message }} [/INST]"#;
````
With those changes the llm was able to generate valid YAML for the examples.

### Quantized models
The llama model can be quantized to reduce the size of the model.
The name of these models will have the size of the quantization in them and also
additional letters. For example:
```
llama-2-7b-chat.Q2_K.gguf         # 2-bit quantization using Q2_K method
llama-2-7b-chat.Q3_K_L.gguf       # 3-bit quantization using Q3_K_L method
llama-2-7b-chat.Q3_K_M.gguf       # 3-bit quantization using Q3_K_M method
llama-2-7b-chat.Q3_K_S.gguf       # 3-bit quantization using Q3_K_S method

llama-2-7b-chat.Q4_0.gguf
llama-2-7b-chat.Q4_K_M.gguf

L = large
M = medium
S = small
```

### LLM_TN (LLM Tensor Names)
Lets take a look at the following line of code:
```c++
const auto tn = LLM_TN(LLM_ARCH_LLAMA);

switch (model.arch) {
    case LLM_ARCH_LLAMA:
    case LLM_ARCH_REFACT:
        model.tok_embd = ml.create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab});
```
Notice that we are calling an operator on `tn`, and specifying a llm_tensor
as the first argument:
```c++
enum llm_tensor {
    LLM_TENSOR_TOKEN_EMBD,
    ...
}
```
And the second argument is a string.

And LLM_TN is defined as:
```c++
  struct LLM_TN {                                                                      
      LLM_TN(llm_arch arch) : arch(arch) {}                                       
                                                                                  
      llm_arch arch;                                                              
                                                                                  
      std::string operator()(llm_tensor tensor, const std::string & suffix) const {
          return LLM_TENSOR_NAMES[arch].at(tensor) + "." + suffix;                
      }                                                                           
      ...
  };
```
In this case that would be like calling:
```c++
LLM_TENSOR_NAMES[LLM_ARCH_LLAMA].at(LLM_TENSOR_TOKEN_EMBD) + "." + "weight";
```
LLM_TEONS_NAMES is defined as:
```c++
static std::map<llm_arch, std::map<llm_tensor, std::string>> LLM_TENSOR_NAMES = {
    {
        LLM_ARCH_LLAMA,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ROPE_FREQS,      "rope_freqs" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_ATTN_ROT_EMBD,   "blk.%d.attn_rot_embd" },
            { LLM_TENSOR_FFN_GATE_INP,    "blk.%d.ffn_gate_inp" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_GATE_EXP,    "blk.%d.ffn_gate.%d" },
            { LLM_TENSOR_FFN_DOWN_EXP,    "blk.%d.ffn_down.%d" },
            { LLM_TENSOR_FFN_UP_EXP,      "blk.%d.ffn_up.%d" },
        },
    },
    ...
};
```
So that would return `token_embd.weight`.
"output_norm.weight"
"output.weight"


### build_llama
This function is used to build the computation graph for the llama model and
is a function that is a member of the `llm_build_context`:
```c++
struct llm_build_context {
    const llama_model    & model;
          llama_context  & lctx;
    const llama_hparams  & hparams;
    const llama_cparams  & cparams;
    const llama_batch    & batch;
    const llama_kv_cache & kv_self;
    ...
    const llm_build_cb & cb;
    ...
    struct ggml_cgraph * build_llama() {
        ...
    }
    ...
}
```
Lets take a closer look at `build_llama`:
```c++
    struct ggml_cgraph * build_llama() {
        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, LLAMA_MAX_NODES, false);

        const int64_t n_embd_head = hparams.n_embd_head_v;
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);
```
`inpL` I think is the input layer. And notice that this is created by calling
`llm_build_inp_embd`. So this is a function that is used by multiple models to
build this input layer. Note that a callback function is also passed in and this
is a member of the `llm_build_context` and the callback is passed into the
constructor. To see where this callback function is defined we have to back up
to llama_build_graph where we have:
```c++
    llm_build_cb cb = [&](struct ggml_tensor * cur, const char * name, int il) {
        if (il >= 0) {
            ggml_format_name(cur, "%s-%d", name, il);
        } else {
            ggml_set_name(cur, name);
        }

        if (!lctx.cparams.offload_kqv) {
            if (strcmp(name, "kqv_merged_cont") == 0) {
                // all nodes between the KV store and the attention output are run on the CPU
                ggml_backend_sched_set_tensor_backend(lctx.sched, cur, lctx.backend_cpu);
            }
        }

        const bool full_offload = lctx.model.n_gpu_layers > (int)lctx.model.hparams.n_layer;
        if (batch.n_tokens < 32 || full_offload) {
            if (il != -1 && strcmp(name, "norm") == 0) {
                for (auto * backend : lctx.backends) {
                    if (ggml_backend_buft_supports_backend(lctx.model.buft_layer[il].buft, backend)) {
                        ggml_backend_sched_set_tensor_backend(lctx.sched, cur, backend);
                        break;
                    }
                }
            }
        }
    };

    struct llm_build_context llm(lctx, batch, cb, worst_case);
    llm.init();

    switch (model.arch) {
        case LLM_ARCH_LLAMA:
            {
                result = llm.build_llama();
            } break;
        case LLM_ARCH_BAICHUAN:
```
So we can see that an llm_build_context is created using the llm_build_cb and
and this is what is passed into llm_build_inp_embd:
```c++
        inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);
```
To see what batch is we have to back up the stack to
llama_new_context_with_model:
```c++
            int n_tokens = (int)std::min(cparams.n_ctx, cparams.n_ubatch);
            int n_past = cparams.n_ctx - n_tokens;
            llama_token token = llama_token_bos(&ctx->model); // not actually used by llama_build_graph, but required to choose between token and embedding inputs graph
            ggml_cgraph * gf = llama_build_graph(*ctx, llama_batch_get_one(&token, n_tokens, n_past, 0), true);
```
Lets inspect some of the variables:
```console
gdb) p n_tokens
$11 = 512
(gdb) p n_past
$12 = 0
(gdb) p token
$13 = 1
(gdb) p ctx->model->vocab.special_bos_id 
$14 = 1

(gdb) p llama_batch_get_one(&token, n_tokens, n_past, 0)
$15 = {n_tokens = 512, token = 0x7fffffff655c, embd = 0x0, pos = 0x0, n_seq_id = 0x0, seq_id = 0x0, 
  logits = 0x0, all_pos_0 = 0, all_pos_1 = 1, all_seq_id = 0}
```
`n_past` is the position in the sequence.

```c++
    if (batch.token) {
        lctx.inp_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, batch.n_tokens);
        cb(lctx.inp_tokens, "inp_tokens", -1);
        ggml_set_input(lctx.inp_tokens);

        inpL = ggml_get_rows(ctx, tok_embd, lctx.inp_tokens);
```
Note that the 1d tensor created is a list of i32 and the size in this case is
512. If we think about the input to the model it is a sequence of tokens which
are the indexes into the vocabulary. So this tensor is a list of tokens.
And the callback will set the name of the tensor to `inp_tokens`. 
Now, the call to ggml_get_rows was something I've not come accross before and
I needed to looking it. I've created a standalone example of how this function
can be used which is in [get-rows.c](../fundamentals/ggml/src/get_rows.c).
So what this is doing is that it extracting rows from `tok_embd` and the rows
to extract are specified by the `lctx.inp_tokens` tensor, which are like
indices. 
```console
(gdb) p ggml_n_dims(tok_embd)
$38 = 2
(gdb) p tok_embd->ne[0]
$39 = 4096
(gdb) p tok_embd->ne[1]
$40 = 32000
```
And I find visualizing it like this helps me understand it better:
```
ne[1]
 |    3
 ↓    2
      0  
      0
      0
           4096

          ne[0] ->
```
And then we have the tensor with the indices:
```console
(gdb) p ggml_n_dims(lctx.inp_tokens)
$43 = 1
(gdb) p lctx.inp_tokens.ne[0]
$44 = 512
```
But keep in mind that this is just building up the computation graph so that
there are no actualy values in the tensors yet, at least not the inp_tokens.

So this makes sense now I think, we have the input tokens which is a list of
indices into the vocabulary. The vocabulary in llama has 32000 tokens and each
token has a an embedding dimention of 4096. What the get rows is doing is that
is it extracting the embeddings for each token in the input.

_wip_



### llm_build_context
This section is doing to take a detailed look at how a computation graph is
build in llama.cpp. I'll be using the `main` example to step through the code.

```console
$ gdb --args ./main -m models/llama-2-7b-chat.Q4_0.gguf -p "What is your name?"
Reading symbols from ./main...
(gdb) br llama.cpp:5810
Breakpoint 1 at 0x4b365f: file llama.cpp, line 5810.
(gdb) r
```

```console
(gdb) f
#4  0x00000000005c08ef in main (argc=5, argv=0x7fffffffc958) at examples/main/main.cpp:199
199	    std::tie(model, ctx) = llama_init_from_gpt_params(params);
```

```c++
std::tuple<struct llama_model *, struct llama_context *> llama_init_from_gpt_params(gpt_params & params) {
    auto mparams = llama_model_params_from_gpt_params(params);

    llama_model * model  = llama_load_model_from_file(params.model.c_str(), mparams);
    if (model == NULL) {
        fprintf(stderr, "%s: error: failed to load model '%s'\n", __func__, params.model.c_str());
        return std::make_tuple(nullptr, nullptr);
    }

    auto cparams = llama_context_params_from_gpt_params(params);

    llama_context * lctx = llama_new_context_with_model(model, cparams);
```
So we load the model from the file system and then setup all the parameters,
like the model parameters and the computation parameters. We pass the model
and the cparams to the `llama_new_context_with_model` function which will now
take a closer look at.

```c++

struct llama_context * llama_new_context_with_model(
                 struct llama_model * model,
        struct llama_context_params   params) {
    ...

    llama_context * ctx = new llama_context(*model);
    // Following this we have a number of fields of the context that are set
    // to the values of the parameters that were passed in.

    ...
    if (!hparams.vocab_only) {
        // initialize backends
```
Backend are initialized here and there are different backends for CUDA, Kompute,
SYCL, Metal, and Vulkan. So the llama_context has members for backends:
```c++
    std::vector<ggml_backend_t> backends;

#ifdef GGML_USE_METAL
    ggml_backend_t backend_metal = nullptr;
#endif

    ggml_backend_t backend_cpu = nullptr;
```
So we have a vector of backends which will initially be empty, optionally if
METAL is used then there is also a backend_metal, and there is also a
backend_cpu.
Notice that they type of the backends is `ggml_backend_t` which is a pointer to
In ggml-alloc.h we have:
```c++
typedef struct ggml_backend * ggml_backend_t;
```
llama.h includes ggml-backend.h.
```c++
    struct ggml_backend {
        ggml_guid_t guid;

        struct ggml_backend_i iface;
        ggml_backend_context_t context;
    };
```
ggml_backend_context_t is a void pointer which I think it to be able to support
different types of backends. The `iface` is a struct of function pointers which
are used to interact with the backend. The `guid` is a unique identifier for the
backend. So lets look a how these backends are initialized.
```c++
#elif defined(GGML_USE_CUBLAS)
        if (model->n_gpu_layers > 0) {
            // with split_mode LLAMA_SPLIT_MODE_NONE or LLAMA_SPLIT_MODE_ROW, only the main GPU backend is used
            if (model->split_mode == LLAMA_SPLIT_MODE_NONE || model->split_mode == LLAMA_SPLIT_MODE_ROW) {
                ggml_backend_t backend = ggml_backend_cuda_init(model->main_gpu);
                if (backend == nullptr) {
                    LLAMA_LOG_ERROR("%s: failed to initialize CUDA%d backend\n", __func__, model->main_gpu);
                    llama_free(ctx);
                    return nullptr;
                }
                ctx->backends.push_back(backend);
            } else {
                // LLAMA_SPLIT_MODE_LAYER requires a backend for each GPU
                for (int device = 0; device < ggml_backend_cuda_get_device_count(); ++device) {
                    ggml_backend_t backend = ggml_backend_cuda_init(device);
                    if (backend == nullptr) {
                        LLAMA_LOG_ERROR("%s: failed to initialize CUDA%d backend\n", __func__, device);
                        llama_free(ctx);
                        return nullptr;
                    }
                    ctx->backends.push_back(backend);
                }
            }
        }
#elif defined(GGML_USE_VULKAN)
  ...
#endif
        ctx->backend_cpu = ggml_backend_cpu_init();
        if (ctx->backend_cpu == nullptr) {
            LLAMA_LOG_ERROR("%s: failed to initialize CPU backend\n", __func__);
            llama_free(ctx);
            return nullptr;
        }
        ctx->backends.push_back(ctx->backend_cpu);
```

```c++
        if (!llama_kv_cache_init(ctx->kv_self, ctx->model, type_k, type_v, kv_size, cparams.offload_kqv)) {
            LLAMA_LOG_ERROR("%s: llama_kv_cache_init() failed for self-attention cache\n", __func__);
            llama_free(ctx);
            return nullptr;
        }
```

### GGML_CALL macro
This macro uses a calling convention of `__ms_abi__` which is the Microsoft ABI
if the `GGML_MULTIPLATFORM` macro is defined:
```c++
#ifdef GGML_MULTIPLATFORM
#    if defined(_WIN32)
#        define GGML_CALL
#    else
#        define GGML_CALL __attribute__((__ms_abi__))
#    endif
#else
#    define GGML_CALL
#endif
```
This initially looked very odd to me as I thought that `__ms_abi__` would only
be used on Windows. But this is done do ensure interoperatibilty beteen
different platforms so we want then to have a common calling convention between
NVCC (NVIDIA CUDA Compiler), ROCm (for AMD GPUs), XCode (for macOS), and the
common GCC and Clang compilers on various platforms.

By defining -DGGML_MULTIPLATFORM during the build, the system enforces that
back-references and function pointers conform to the Microsoft ABI (__ms_abi__).
This conformance is crucial for ensuring compatibility and predictable behavior
when the system interfaces with GPU modules compiled with tools that support the
MS ABI, such as NVCC (NVIDIA CUDA Compiler), ROCm (for AMD GPUs), XCode
(for macOS), and the common GCC and Clang compilers on various platforms.
