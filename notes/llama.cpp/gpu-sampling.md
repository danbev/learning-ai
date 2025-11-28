### GPU Sampling
Currently, sampling is implemented on the CPU in llama.cpp through the sampling
chain configured. After `llama_decode` is called the logits are passed to the
sampler chain which can modify the logits and probabilities. For example this
is done in `common::set_logits`:
```c++
struct common_sampler {
    common_params_sampling params;

    struct llama_sampler * grmr;
    struct llama_sampler * chain;

    ring_buffer<llama_token> prev;

    std::vector<llama_token_data> cur;

    llama_token_data_array cur_p;

    void set_logits(struct llama_context * ctx, int idx) {
        const auto * logits = llama_get_logits_ith(ctx, idx);

        const llama_model * model = llama_get_model(ctx);
        const llama_vocab * vocab = llama_model_get_vocab(model);

        const int n_vocab = llama_vocab_n_tokens(vocab);

        cur.resize(n_vocab);

        for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
            cur[token_id] = llama_token_data{token_id, logits[token_id], 0.0f};
        }

        cur_p = { cur.data(), cur.size(), -1, false };
    }
```
And the data structure that is passed to the samplers looks like this:
```c++
    typedef struct llama_token_data {
        llama_token id; // token id
        float logit;    // log-odds of the token
        float p;        // probability of the token
    } llama_token_data;

    typedef struct llama_token_data_array {
        llama_token_data * data;
        size_t size;
        int64_t selected; // this is the index in the data array (i.e. not the token id)
        bool sorted;      // note: do not assume the data is sorted - always check this flag
    } llama_token_data_array;
```
`cur_p` is the struct that is passed through all the samplers in the chain, and
samplers can modify the logits, probabilities, sort, and modifiy the size of the
array. The GPU samplers should be able to do the same. 

So we have array of elements and for each element stored we have the token id, the
logits for the token and optionally the probability for the token. I say probably
as the probabilities are intially 0.0f and is something that samplers in the chain
can calculate and modify. These have to be recalculated if the logits are modified in
any way (like also if the size of the array is modified).

To be processed by a GGML graph the data, the information in `cur_p` needs to be
in the form of a `ggml_tensor` (or multiple).


### Current implementation
In contrast to CPU sampling where the sampling operations are performed after
the models graph has been executed, GPU sampling is part of the same execution
graph. All of the sampling can be done on the GPU or parts of it can be done on
the GPU and the rest on the CPU.

#### Configuration of GPU samplers
GPU samplers are configured before the context is created and a GPU sampler
can be configured per sequence:
```console
    struct llama_sampler_chain_params params = llama_sampler_chain_default_params();
    struct llama_sampler * samplers = llama_sampler_chain_init(params);

    llama_sampler_chain_add(samplers, llama_sampler_gpu_init_greedy());

    std::vector<llama_sampler_seq_config> gpu_sampler_configs = {
        { 0, gpu_sampler_chain }
    };
```
The above is only showing one sampler but multiple samplers can be added to the
gpu_samplers_config vector.

These samplers are then passed into the context parameters when creating the
context:
```c++
    llama_context_params cparams = llama_context_default_params();
    cparams.samplers = sampler_configs.data();
    cparams.n_samplers = sampler_configs.size();

    ctx = llama_init_from_model(model, cparams);
```

When the model graph is built the GPU samplers will called to enable them to
add their operations to the graph:
```c++
ggml_cgraph * llama_model::build_graph(const llm_graph_params & params) const {
    std::unique_ptr<llm_graph_context> llm;
    ...

    // add GPU sampling layers (if any)
    llm->build_sampling(*this, params);
```
The llama_sampler_i interface as been extended with 4 new methods in the API, 
and they are currently all named with a `_ggml` suffix to indicate that they
are for GPU sampling:
```c++
        void                   (*init_ggml)(struct llama_sampler      * smpl,
                                            ggml_backend_buffer_type_t  buft);

        void                   (*set_input_ggml)( struct llama_sampler * smpl,
                                                       ggml_context * ctx,
                                                        ggml_cgraph * gf);

        void                   (*apply_ggml)(  struct llama_sampler * smpl,
                                                       ggml_context * ctx,
                                                        ggml_cgraph * gf,
                                            llama_sampler_ggml_data * ggml_data);

        void                   (*accept_ggml)( struct llama_sampler * smpl,
                                                       ggml_context * ctx,
                                                        ggml_cgraph * gf,
                                               struct ggml_tensor * selected_token);

```
The _init_ggml_ function allows GPU samplers to create input tensors that they
might need. The ggml_backend_buffer_type should be used so that the tensors are
created using this backend buffer type, which is the same as the ouput logits
backend. This avoids splits in the computation graph that would require data
transfer between different backends.
For example:
```c++
struct llama_sampler_gpu_dist_ctx {
    const uint32_t seed;
          uint32_t seed_cur;
    std::mt19937   rng;

    struct ggml_tensor * uniform;
    struct ggml_context * ctx;
    ggml_backend_buffer_t buffer;
};

static void llama_sampler_gpu_dist_init_ggml(
        struct llama_sampler      * smpl,
        ggml_backend_buffer_type_t  buft) {

    auto * sctx = (llama_sampler_gpu_dist_ctx *) smpl->ctx;
    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    sctx->ctx = ggml_init(params);

    // Create the uniform random scalar input tensor. This will be set by
    // llama_sampler_gpu_dist_set_input_ggml after this graph is built.
    sctx->uniform = ggml_new_tensor_1d(sctx->ctx, GGML_TYPE_F32, 1);
    ggml_set_name(sctx->uniform, "uniform");
    ggml_set_input(sctx->uniform);
    ggml_set_output(sctx->uniform);

    // Allocate all tensors from our context to the backend
    sctx->buffer = ggml_backend_alloc_ctx_tensors_from_buft(sctx->ctx, buft);
}
```

The _set_input_ggml_ function is called after the computation graph has been
scheduled but before it is computed. This allows the GPU sampler to set any
input for the tensors it created in init_ggml.
```c++
static void llama_sampler_gpu_dist_set_input_ggml(struct llama_sampler * smpl) {
    auto * sctx = (llama_sampler_gpu_dist_ctx *) smpl->ctx;
    GGML_ASSERT(sctx->uniform != nullptr);

    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    const float rnd = dist(sctx->rng);
    ggml_backend_tensor_set(sctx->uniform, &rnd, 0, sizeof(float));
}
```

The _apply_ggml_ function is where the GPU sampler adds its operations to the
graphs. When the graph is built, the configured sampler's _apply function is
called which allows them to add operations/nodes to the computation graph.

The _accept_ggml_ functions allows GPU samplers to update their tensor states if needed.

```c++
static void llama_sampler_gpu_greedy_apply_ggml(
        struct llama_sampler           * smpl,
        struct ggml_context            * ctx,
        struct ggml_cgraph             * gf,
        struct llama_sampler_ggml_data * ggml_data) {
    (void) gf;
    auto * sctx = (llama_sampler_gpu_greedy_ctx *) smpl->ctx;

    struct ggml_tensor * argmax_result = ggml_argmax(ctx, ggml_data->logits);
    ggml_set_name(argmax_result, "argmax_result");
    ggml_backend_sched_set_tensor_backend(sctx->sched, argmax_result, sctx->backend);
    ggml_data->sampled_token = argmax_result;
}
```
And here we also see the usage of the scheduler and backend to ensure that the
tensor is created on the correct backend.

accept_ggml is called after the GPU graph has been executed to allow the GPU
sampler to accept the selected token and update its state. Note that currently
no GPU samplers maintain any state in this way and is something that needs more
work.

set_input_ggml is called after the computation graph has been schduled but before
it is computed. This allows the GPU sampler to set any input. This is currently
used by the temp sampler to set a random number tensor that is used for sampling.

Support has been added to llama-cli and llama-server to enable testing of the GPU
sampling features. Even though the implementation might still change and perhaps
significantly it was valuable to implement that support to see how this would work
and it uncovered some isseus that the tests missed.

The pull request can be found here:
https://github.com/ggml-org/llama.cpp/pull/17004


#### Setting/unsetting a GPU sampler
Currently the samplers are configured for a specific sequence id which happens
at the same time that the context is created.
```c++
    std::unordered_map<llama_seq_id, llama_sampler*> samplers;
```
In the llama_context constructor we have the following:
```c++
llama_context::llama_context(
        const llama_model & model,
              llama_context_params params) :
    model(model),
    balloc(std::make_unique<llama_batch_allocr>(model.hparams.n_pos_per_embd())) {
    ...

    // GPU samplers
    if (params.samplers != nullptr && params.n_samplers > 0) {
        samplers.reserve(params.n_samplers);

        for (size_t i = 0; i < params.n_samplers; ++i) {
            const auto & config = params.samplers[i];
            samplers[config.seq_id] = config.sampler;
        }
    }
```
Now, we might want to unset or change the GPU sampler for a specific sequence.
Unsetting would just be clearing that so lets start with that functionality.

----

The sections below contains some notes taken during the initial design and
exporation of GPU sampling llama.cpp and are not really relavant anymore.

### Suggested approach
One way could be to store the tensors in a struct like this:
```c++
    struct llama_sampler_ggml_data {
        struct ggml_tensor * ids;      // [n_vocab] - GGML_TYPE_I32
        struct ggml_tensor * logits;   // [n_vocab] - GGML_TYPE_F32
        struct ggml_tensor * probs;    // [n_vocab] - GGML_TYPE_F32
        struct ggml_tensor * selected; // [1]       - GGML_TYPE_I32
        struct ggml_tensor * size;     // [1]       - GGML_TYPE_I32  number of valid tokens in the arrays (<= n_vocab)
        struct ggml_tensor * sorted;   // [1]       - GGML_TYPE_I32  whether data is sorted by logits/probs
    };
};
```
The token ids can be stored as `I32` type tensors, and the logits and
probabilities as `F32`.

This would allow a function declarations to look something like this:
```c++
        void                   (*apply_ggml)(  struct llama_sampler * smpl,
                                                       ggml_context * ctx,
                                                        ggml_cgraph * gf,
                                            llama_sampler_ggml_data * ggml_data);

        void                   (*accept_ggml)( struct llama_sampler * smpl,
                                                       ggml_context * ctx,
                                                        ggml_cgraph * gf,
                                               struct ggml_tensor * selected_token);
```
This way multiple GPU samplers can be chained together and they can all update
the graph with the operations they need to perform.

To be able to perform the GPU sampling operations on the GPU this will be done
in a similar manner to how pooling is currently applied. To enable this a
llama_sampler has been added to llama_context_params:
```c++
        struct llama_sampler_chain_params gpu_sampler_params = llama_sampler_chain_default_params();
        struct llama_sampler * gpu_sampler_chain = llama_sampler_chain_init(gpu_sampler_params);
        llama_sampler_chain_add(gpu_sampler_chain, llama_sampler_gpu_init_greedy());

        llama_context_params cparams = llama_context_default_params();
        cparams.sampler = sampler;

        auto ctx = llama_init_from_model(model, cparams);
```
When the models graph is built we will then also build the sampling graph:
```c++
ggml_cgraph * llama_model::build_graph(const llm_graph_params & params) const {
    std::unique_ptr<llm_graph_context> llm;
    ...

    // add on pooling layer
    llm->build_pooling(cls, cls_b, cls_out, cls_out_b);

    // add GPU sampling layers (if any)
    llm->build_sampling(*this);
    ...
}
```
All the sampler will be applied that have been configured. Depending on that
what sampler actually does, like it may just modify the logits like a temperature
sampler, or it may filter the logits like a top-k sampler, or calculate the
probabilities like a dist sampler. The sampler could select a token directly
like a greedy sampler so it would be possible to skip and CPU sampler and just
run GPU samplers.

To get the probabilites generated:
```c++
    float * probs = llama_get_sampled_probs(test_ctx.ctx);
```
To get the selected token:
```c++
    llama_token id = llama_get_sampled_token(test_ctx.ctx);
```

But it is also possible to have CPU samplers after the normal llama_decode call
which will be able to operate on the logits just like before but this opens up
the possiblility to mix GPU samplers and CPU samplers, for example running
temperature scaling and top-k on the GPU and then dist and greedy on the CPU.

### GPU Sampler state
This was brought up in the feedback and something that we need to consider. The
parameters to the GPU samplers can work as the currently do for CPU samplers and
allows the GPU sampler to use them, for example this is what the top-k sampler
does with the 'k' parameter.

But the for something like the dist sampler that needs to maintain the RNG state
I'm not exactly sure how to handle. There is a difference here as the GPU
samplers when their apply_ggml function is called they build/add to the
computation graph that will later be run. But the CPU sampler actually perform
their operations directly in this function on the CPU. And this is where the
rng state for the dist CPU sampler is used.

Thinking about this some more, perhaps the GPU sampler should only to the
expensive filtering on the GPU like top-k, top-p, min-p etc. And then with the
reduced set of logits/probabilites the CPU samplers can then take over for things
like dist, greedy, mirostate, penalties, and grammer. This would simplify the
GPU samplers as they would not require additional state to be maintained.


### Feedback/Questions
* The GPU sampling ops should probably be operating on constant-shape tensors.
  We want to have static graphs for efficiency.
The current suggestion that I proposed above uses dynamic tensors, the samplers
can change the sized of them. This is an issue because each time the graph
structure changes the graph needs to be rebuilt. So instead of having:
```c++
    struct llama_sampler_ggml_data {
        struct ggml_tensor * ids;     // [n_vocab] - GGML_TYPE_I32
        struct ggml_tensor * logits;  // [n_vocab] - GGML_TYPE_F32
        struct ggml_tensor * probs;   // [n_vocab] - GGML_TYPE_F32
        int64_t size;                 // number of valid tokens in the arrays (<= n_vocab)
        int64_t selected;             // index in the array (-1 if not yet selected)
        bool sorted;                  // whether data is sorted by logits/probs
    };
```
This way the tensors will have a fixed tensor size.

* The conversion from llama_token_data_array to llama_sampler_ggml_data should not
  be performed when we are using only GPU samplers. Otherwise it would incur
  significant data transfer to negate the benefits of GPU sampling.

* It was also brought up that CPU samplers may require addition tensors for storing
  parameters and state. These tensors need to be preallocated and made availabe to the
  GPU samplers in some way. An example of such a sampler is the dist sampler that needs
  to store the RNG (random number generator) state.

* Does it make sense to implement all the current CPU samplers on the GPU?  
Would it perhaps be enough to implement the samplers like temperature, top-k,
top-p, min-p on the GPU so that they can take advantage of the logits already
being on the GPU, then filter down the logits/probabilities before copying them
from device to system memory for the remaining CPU samplers to process?
So perhaps we could start by implementing:
- Temparature
- Top-p
- Min-p
- additional?

### Top-k GPU sampling
So currently we have a sampler chain per sequence, and each sampler is provided
with the logits for its sequence.

So in our sampler we have:
```console
(gdb) p ggml_data->logits->ne
$9 = {32000, 1, 1, 1}
```
These are the logits for the last token in the sequence, so we have 32000 tokens.

So going through this. top_k will look like this:
```console
(gdb) p top_k->ne
$3 = {8, 1, 1, 1}
```
This is just one row with the indices of the top 8 logits.

Then we reshape the logits to become 32000 rows each with one element.
And we use ggml_get_rows to select those values using the indices which produces
```console
  (gdb) p top_k_rows->ne
  $6 = {1, 8, 1, 1}
```
And each or these rows contains a token, and the first row is the top selection,
followed by the second etc.

_wip_


### Dist GPU sampling
To implement dist sampling on the GPU we need to be able to generate random
and uniform numbers on the GPU. I don't think that  GGML currently has support
for generating random numbers nor that GPU backend have such an operation.
But instead what we could do is that we enable the GPU sampler's _apply_ggml
function to create a tensor in the samplers context. And we then add a new
function to the sampler interface named set_input_ggml. This function will be
called after the graph has been built and scheduled but before it has been
executed. This way samplers like this one can generate the random numbers on
the CPU and then upload them to the GPU before the graph is executed. This
involved some data transfer but only of a relatively small tensor of random
numbers.

I naively just created the tensor in the apply_ggml function like this:
```c++
    // Create the uniform random scalar input tensor. This will be set by
    // llama_sampler_gpu_dist_set_input_ggml after this graph is built, but
    // before it is executed.
    struct ggml_tensor * uniform = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    sctx->uniform = uniform;
    ggml_set_name(uniform, "uniform");
    ggml_set_input(uniform);
    ggml_set_output(uniform);
```
Now, if we think about how samplers are used, then are added to then end of the
models graphs, after the ouput tensors. So it makes sense to create the samplers
on the same backend as that tensor so that there is not copying between the
backends.

So we need to do something like the following when creaing tensors in GPU 
samplers:
```c++
  ggml_backend_sched_set_tensor_backend(sched, uniform, target_backend);
```
Notice
```console
(gdb) p *model->pimpl->dev_output.dev
```
```c++
    GGML_API void ggml_backend_sched_set_tensor_backend(ggml_backend_sched_t sched,
        struct ggml_tensor * node, ggml_backend_t backend);
    GGML_API ggml_backend_t ggml_backend_sched_get_tensor_backend(ggml_backend_sched_t sched,
        struct ggml_tensor * node);
```

So perhaps we can add a function the samplers interface that sets this information,
the scheduler and the backend tensor to use. I'll try this out and see if it
works and how it "feels". The samplers that need to maintain states or create
tenorsr would need to implement this function and also add members for the
scheduler and the target backend.


### llama-server
Now GPU sampling for llama-cli was pretty straightforward as there is bacially
just one GPU sampler needed to be configured. And recall that the samplers need
to be configured before the context is created as the GPU samplers add to the
models computation graph and are not something that is processed after the model
graph like CPU samplers. 

It is possible to have a sampler per sequence in llama.cpp, and llama-server
support multiple slots/sequences. And it is possible to requests to specify a
specific slot to be used. So we could provide a configuration option for
llama-server to have different gpu sampler chains per slot.

```console
-gpu-sampling                          enable GPU sampling (default: disabled)
--gpu-top-k N                           GPU top-k sampling (default: 40, <= 0 = disabled)
--gpu-top-p-approx-k N                  GPU top-p approximation using top-k (default: 0, 0 = disabled)
--gpu-temp N                            GPU temperature (default: 0.80, 0.0 = disabled, greedy sampling)
--gpu-softmax                           add GPU softmax to sampling chain (default: disabled)
--gpu-dist                              add GPU dist (final sampling) to sampling chain (default: disabled)
--gpu-slot SLOT_ID:CONFIG               configure GPU sampling for a specific slot (server only)
                                        format: SLOT_ID:top_k=N,temp=F,dist=BOOL
                                        example: --gpu-slot 0:top_k=20,temp=0.8,dist=true --gpu-slot
                                        1:top_k=40,temp=0.5
```
And this could then be use as follows:
```console
./build-gpu-sampling/bin/llama-server \
      -m models/Qwen2.5-VL-3B-Instruct-Q8_0.gguf \
      --gpu-sampling \
      --gpu-top-k 20 \
      --gpu-temp 0.8 \
      --gpu-dist \
      -ngl 99 \
      -v
```

### Mixed backend and cpu sampling
I initially developed this as an all or nothing, either we perform backend
sampling for the batch, or we perform CPU sampling for the batch (after
llama_decode for CPU samplers and backend samplers are part of the models graph).

But we need to support the case where we have mixed sequences in a batch. What I
mean is that may have configured a backend sampler for sequence 0 but just that
one and our batch might contains two sequences. In this case we need to be able
to perform the backend sampling for sequence 0 and CPU sampling for sequence 1.

For example, the following batch is used by the test_backend_cpu_mixed_batch:
```
n_tokens: 4
token[0]: tok=1    , pos=0, n_seq_id=1, seq_ids=[0], logits=0
token[1]: tok=15043, pos=1, n_seq_id=1, seq_ids=[0], logits=1
token[2]: tok=1    , pos=0, n_seq_id=1, seq_ids=[1], logits=0
token[3]: tok=3834 , pos=1, n_seq_id=1, seq_ids=[1], logits=1
```
This is a batch with 4 tokens, two sequences (seq_id 0 and seq_id 1), each with
two tokens. Sequence 0 has a backend sampler configured (a greedy sampler) and
```console
(gdb) p sampling.samplers
$17 = std::unordered_map with 1 element = {[0] = 0x555555f5ab00}
```
In this case we need to ensure that there is pinned memory allocated for the
backend sampler and for the CPU sampler (which needs the logits only).

I run into an issue with the original implementation using the llama-server and
the webui where decoding with backend samplers works but after changing to cpu
sampling, this could cause a slot(sequence) to still be using a backend sampler
which would mean that the logits in output_reserve would not be allocated and
there would be an error.

The suggested solution for this is to inspect the batch and see what types of
output are needed, I mean if a token in the batch is set as output and that
sequence has a backend sampler configure we need to ensure that the output_reserve
is allocated. And if it does not we have to ensure that logits are allocated
and are copied back to the host.

One issue with my initial implementation was that the backend samplers I just had
two cases and allowed the cpu sampling case to just work as it currently does,
but this means that when we have mixed sampling we will be copying over all
the logits to the host even for sequences that have backend samplers configured.

One solution is to pass the batch into output_reserve so that it can inspect the
batch and see what types of outputs are needed. This way it can allocate only
what is needed. This is what I have implemented in the PR now.
