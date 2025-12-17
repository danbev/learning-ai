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

### Updated backend sampling
A had a few weeks away from this work while working on another task but this
section will do through the updates that have been made in the mean time.

llama-cli has been replaced with a more interactive cli and the old version is
now named llama-completion which is what we will be looking at here. The first
interesting line is the following:

```c++
    auto llama_init = common_init_from_params(params);
```
```c++
common_init_result_ptr common_init_from_params(common_params & params);

using common_init_result_ptr = std::unique_ptr<common_init_result>;
```

unit_result looks like this:
```c++
struct common_init_result {
    common_init_result(common_params & params);
    ~common_init_result();

    llama_model * model();
    llama_context * context();
    common_sampler * sampler(llama_seq_id seq_id);

    std::vector<llama_adapter_lora_ptr> & lora();

    void free_context();

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
```
Notice that there is a sampler method that takes a sequence id and returns the
sampler for that sequence which we will review later.

Next we create a new common_init_result and use it with the unique pointer
constructor. I wonder if we should prefer make_unique here instead which cannot
throw after allocating the memory?
```c++
common_init_result_ptr common_init_from_params(common_params & params) {
    common_init_result_ptr res(new common_init_result(params));
```
So lets take a look at the constructor:
```c++
common_init_result::common_init_result(common_params & params) : pimpl(new impl{}) {
```
We can see that this will create a new unique_ptr (pimpl) which is a pimpl idiom.
```c++
    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), mparams);
    if (model == NULL) {
        return;
    }

    pimpl->model.reset(model);
```
The implementation struct looks like this:
```c++
struct common_init_result::impl {
    impl() = default;
    ~impl() = default;

    llama_model_ptr   model;
    llama_context_ptr context;

    std::vector<llama_adapter_lora_ptr> lora;

    std::vector<common_sampler_ptr> samplers;
    std::vector<llama_sampler_seq_config> samplers_seq_config;
};
```
So it has a model, a context, lora, samplers, and sampler sequence configs which
look the same as before but this struct is completely new.

The reset is resetting the model unique pointer nothing else.

Next we have the following which is related to sampling, and recall that we
are still in common_init_result:
```c++
    // updates params.sampling
    // TODO: fix naming
    common_init_sampler_from_model(model, params.sampling);
```
```c++
// TODO: move to common/sampling
static void common_init_sampler_from_model(
    const llama_model * model,
    common_params_sampling & sparams) {

    const uint64_t config = sparams.user_sampling_config;
```
user_sampling_config is defined as:
```c++
    uint64_t user_sampling_config = 0; // bitfield to track user-specified samplers
```

Next, a few lambda functions are defined:
```c++
    auto get_int32 = [&](const char * key, int32_t & dst, uint64_t user_config) {
        // bitwise and to check if the user has already set this manually, if
        // so skip.
        if (config & user_config) {
            return;
        }

        char buf[64] = {0};
        // get value from model metadata to see if it has a recommended value 
        // for this key.
        if (llama_model_meta_val_str(model, key, buf, sizeof(buf)) > 0) {
            char * end = nullptr;
            int32_t v = strtol(buf, &end, 10);
            if (end && end != buf) {
                dst = v;
            }
        }
    };

    auto get_float = [&](const char * key, float & dst, uint64_t user_config) {
        if (config & user_config) {
            return;
        }

        char buf[128] = {0};
        if (llama_model_meta_val_str(model, key, buf, sizeof(buf)) > 0) {
            char * end = nullptr;
            float v = strtof(buf, &end);
            if (end && end != buf) {
                dst = v;
            }
        }
    };
```
```c++
enum common_params_sampling_config : uint64_t {
    COMMON_PARAMS_SAMPLING_CONFIG_SAMPLERS        = 1 << 0,
    COMMON_PARAMS_SAMPLING_CONFIG_TOP_K           = 1 << 1,
    COMMON_PARAMS_SAMPLING_CONFIG_TOP_P           = 1 << 2,
    COMMON_PARAMS_SAMPLING_CONFIG_MIN_P           = 1 << 3,
    COMMON_PARAMS_SAMPLING_CONFIG_XTC_PROBABILITY = 1 << 4,
    COMMON_PARAMS_SAMPLING_CONFIG_XTC_THRESHOLD   = 1 << 5,
    COMMON_PARAMS_SAMPLING_CONFIG_TEMP            = 1 << 6,
    COMMON_PARAMS_SAMPLING_CONFIG_PENALTY_LAST_N  = 1 << 7,
    COMMON_PARAMS_SAMPLING_CONFIG_PENALTY_REPEAT  = 1 << 8,
    COMMON_PARAMS_SAMPLING_CONFIG_MIROSTAT        = 1 << 9,
    COMMON_PARAMS_SAMPLING_CONFIG_MIROSTAT_TAU    = 1 << 10,
    COMMON_PARAMS_SAMPLING_CONFIG_MIROSTAT_ETA    = 1 << 11,
};
```
```c++
    // Sampling sequence string of sampler chain to use and the order
    // same check using a bitwise and to see if the user has set this manually.
    if (!(config & common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_SAMPLERS)) {
        char buf[512] = {0};

        if (llama_model_meta_val_str(model, llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_SEQUENCE), buf, sizeof(buf)) > 0) {
            // the value of the field is a semicolon separated list of sampler names
            const std::vector<std::string> sampler_names = string_split<std::string>(std::string(buf), ';');
            if (!sampler_names.empty()) {
                // set the samplers from the model
                sparams.samplers = common_sampler_types_from_names(sampler_names, true);
            }
        }
    }
```
```c++
    enum llama_model_meta_key {
        LLAMA_MODEL_META_KEY_SAMPLING_SEQUENCE,
        LLAMA_MODEL_META_KEY_SAMPLING_TOP_K,
        LLAMA_MODEL_META_KEY_SAMPLING_TOP_P,
        LLAMA_MODEL_META_KEY_SAMPLING_MIN_P,
        LLAMA_MODEL_META_KEY_SAMPLING_XTC_PROBABILITY,
        LLAMA_MODEL_META_KEY_SAMPLING_XTC_THRESHOLD,
        LLAMA_MODEL_META_KEY_SAMPLING_TEMP,
        LLAMA_MODEL_META_KEY_SAMPLING_PENALTY_LAST_N,
        LLAMA_MODEL_META_KEY_SAMPLING_PENALTY_REPEAT,
        LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT,
        LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT_TAU,
        LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT_ETA,
    };
```
Next we will go through all the rest of the sampling parameters:
```c++
    get_int32(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_TOP_K),           sparams.top_k,           common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_TOP_K);
    get_float(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_TOP_P),           sparams.top_p,           common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_TOP_P);
    get_float(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_MIN_P),           sparams.min_p,           common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_MIN_P);
    get_float(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_XTC_PROBABILITY), sparams.xtc_probability, common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_XTC_PROBABILITY);
    get_float(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_XTC_THRESHOLD),   sparams.xtc_threshold,   common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_XTC_THRESHOLD);
    get_float(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_TEMP),            sparams.temp,            common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_TEMP);
    get_int32(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_PENALTY_LAST_N),  sparams.penalty_last_n,  common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_PENALTY_LAST_N);
    get_float(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_PENALTY_REPEAT),  sparams.penalty_repeat,  common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_PENALTY_REPEAT);
    get_int32(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT),        sparams.mirostat,        common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_MIROSTAT);
    get_float(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT_TAU),    sparams.mirostat_tau,    common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_MIROSTAT_TAU);
    get_float(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT_ETA),    sparams.mirostat_eta,    common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_MIROSTAT_ETA);
}
```
So that functio is all about setting sampling parameters from the models.

Back in common_init_result we have:
```c++
    if (params.sampling.ignore_eos && llama_vocab_eos(vocab) == LLAMA_TOKEN_NULL) {
        LOG_WRN("%s: warning: vocab does not have an EOS token, ignoring --ignore-eos\n", __func__);
        params.sampling.ignore_eos = false;
    }
```
params.sampling.ignore_eos = true tells the sampler that even if it sees the EOS
token, don't stop, just pick the next most likely token and keep going.
But if the models vocab does not even have an EOS token this setting does not
make sense so we warn the user and disable it.

Next we go through all the tokens in the vocabulary and and where it finds stop
tokens is set the logit bias for that token to -INFINITY:
```c++
    // initialize once
    for (llama_token i = 0; i < llama_vocab_n_tokens(vocab); i++) {
        // if end of generation token, set logit bias to -infinity
        if (llama_vocab_is_eog(vocab, i)) {
            LOG_INF("%s: added %s logit bias = %f\n", __func__, common_token_to_piece(vocab, i).c_str(), -INFINITY);
            params.sampling.logit_bias_eog.push_back({i, -INFINITY});
        }
    }

    if (params.sampling.ignore_eos) {
        // add EOG biases to the active set of logit biases
        params.sampling.logit_bias.insert(
                params.sampling.logit_bias.end(),
                params.sampling.logit_bias_eog.begin(),
                params.sampling.logit_bias_eog.end());
    }
```
Next we resize the samplers, and their configurations to the number of sequences
that can be active:
```c++
    // init the backend samplers as part of the context creation
    pimpl->samplers.resize(cparams.n_seq_max);
    pimpl->samplers_seq_config.resize(cparams.n_seq_max);
```
Next for each sequence we initialize the sampler and set its configuration:
```c++
    for (int i = 0; i < (int) cparams.n_seq_max; ++i) {
        pimpl->samplers[i].reset(common_sampler_init(model, params.sampling));
        pimpl->samplers_seq_config[i] = { i, common_sampler_get(pimpl->samplers[i].get()) };
    }
```
Recall that samplers is a vector of unique pointers to and we are calling
common_sampler_init in common/sampling.cpp. And recall the samplers_seq_config
if just a mapping of a sequence id to a sampler chain.
```c++
struct common_sampler * common_sampler_init(const struct llama_model * model, const struct common_params_sampling & params) {
    const llama_vocab * vocab = llama_model_get_vocab(model);

    llama_sampler_chain_params lparams = llama_sampler_chain_default_params();

    lparams.no_perf = params.no_perf;

    llama_sampler * chain = llama_sampler_chain_init(lparams);

    bool grammar = false;
    std::vector<llama_sampler *> samplers;
    ...
    if (params.has_logit_bias()) {
        samplers.push_back(
            llama_sampler_init_logit_bias(llama_vocab_n_tokens(vocab), params.logit_bias.size(), params.logit_bias.data()));
    }
```
And then the rest of the CPU samplers are added to the chain.

So the above was adding the CPU samplers to the chain but no GPU samplers have
been added yet.
Next if backend sampling is enabled then we pass the samplers to the context
params so that they can be used in the llama-context constructor where they are
needed.
```c++
    // TODO: temporarily gated behind a flag
    if (params.sampling.backend_sampling) {
        cparams.samplers   = pimpl->samplers_seq_config.data();
        cparams.n_samplers = pimpl->samplers_seq_config.size();
    }

    llama_context * lctx = llama_init_from_model(model, cparams);
```
Notice that the context parameters samplers is using the same samplers that
were created above for the CPU samplers.

```c++
    try {
        auto * ctx = new llama_context(*model, params);
        return ctx;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: failed to initialize the context: %s\n", __func__, err.what());
    }
```
```c++
llama_context::llama_context(
        const llama_model & model,
              llama_context_params params) :
    model(model),
    balloc(std::make_unique<llama_batch_allocr>(model.hparams.n_pos_per_embd())) {
    ...

    // Initialize backend samplers here so they are part of the sampling graph
    // before the reserve passes run later in this function. This avoids a later
    // re-reserve when graph nodes change.
    if (params.samplers != nullptr && params.n_samplers > 0) {
        for (size_t i = 0; i < params.n_samplers; ++i) {
            const auto & config = params.samplers[i];

            if (llama_sampler_chain_get(config.sampler, -1) == nullptr) {
                throw std::runtime_error("the backend samplers must be of type llama_sampler_chain");
            }

            if (set_sampler(config.seq_id, config.sampler)) {
                const int n_samplers = llama_sampler_chain_n(config.sampler);

                LLAMA_LOG_INFO("%s: setting backend sampler for seq_id %d (n = %d)\n", __func__, config.seq_id, n_samplers);
            }
        }
    }

```
set_sampler is new to me:
```c++
bool llama_context::set_sampler(llama_seq_id seq_id, llama_sampler * sampler) {
    LLAMA_LOG_DEBUG("%s: seq_id = %d, sampler = %p\n", __func__, (int) seq_id, (void *) sampler);

    // So the sampler must not be null, must have the backend functions, and
    // must have a non-zero number of samplers in the chain.
    const bool can_offload =
        sampler &&
        sampler->iface->backend_init &&
        sampler->iface->backend_apply &&
        llama_sampler_chain_n(sampler) > 0;

    if (sampler && can_offload) {
        // get the backend buffer type of the models output device.
        ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(model.dev_output());
        auto * host_buft = ggml_backend_dev_host_buffer_type(model.dev_output());
        if (host_buft) {
            buft = host_buft;
        }

        // initialize the sampler for backend sampling
        sampler->iface->backend_init(sampler, buft);

        sampling.samplers[seq_id] = sampler;

        return true;
    }

    if (sampler && !can_offload) {
        LLAMA_LOG_WARN("%s: sampler '%s' for seq_id = %d, cannot be offloaded to the backend\n", __func__, llama_sampler_name(sampler), seq_id);
        sampling.samplers.erase(seq_id);
        return false;
    }

    sampling.samplers.erase(seq_id);

    return true;
}
```
Notice that this is where `backend_init` is called on the samplers.

All backend samplers inherit from llama_sampler_backend:
```c++
struct llama_sampler_logit_bias : public llama_sampler_backend {
    const int32_t n_vocab;

    const std::vector<llama_logit_bias> logit_bias;

    std::vector<llama_logit_bias> to_search;

    struct ggml_tensor * inp_logit_bias;
    struct ggml_tensor * inp_logit_idxs;

    ggml_context_ptr        inp_ctx;
    ggml_backend_buffer_ptr inp_buf;
};
```
```c++
// common backend sampler functionality
//
// +name : means that the sampler is supported and will run on the backend
// -name : means that a ggml operator is not supported by the backend
//
struct llama_sampler_backend {
    llama_sampler_backend(const char * name) : name(name), name_ext(name), is_init(false), support(false) {}

    const char * get_name() {
        if (!is_init) {
            return name.c_str();
        }

        if (support) {
            name_ext = "+" + name;
        } else {
            name_ext = "-" + name;
        }

        return name_ext.c_str();
    }

    void init(bool support) {
        GGML_ASSERT(this->is_init == false);

        this->is_init = true;
        this->support = support;
    }

private:
    std::string name;
    std::string name_ext;

    bool is_init;
    bool support;
};
```
So lets take a look at the llama_sampler_logit_bias backend init function:
```c++
static bool llama_sampler_logit_bias_backend_init(
        struct llama_sampler       * smpl,
        ggml_backend_buffer_type_t   buft) {
    auto * sctx = (llama_sampler_logit_bias *) smpl->ctx;

    // This will set is_init and support to true
    sctx->init(true);

    if (sctx->logit_bias.empty()) {
        return true;
    }

    ggml_init_params params = {
        /*.mem_size   =*/ 2*ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    sctx->inp_ctx.reset(ggml_init(params));

    const size_t n = sctx->logit_bias.size();

    sctx->inp_logit_bias = ggml_new_tensor_2d(sctx->inp_ctx.get(), GGML_TYPE_F32, 1, n);
    ggml_set_name(sctx->inp_logit_bias, "logit_bias");
    ggml_set_input(sctx->inp_logit_bias);

    sctx->inp_logit_idxs = ggml_new_tensor_1d(sctx->inp_ctx.get(), GGML_TYPE_I32, n);
    ggml_set_name(sctx->inp_logit_idxs, "logit_idxs");
    ggml_set_input(sctx->inp_logit_idxs);

    // Allocate all tensors from our context to the backend
    sctx->inp_buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(sctx->inp_ctx.get(), buft));

    return true;
}
```
Notice that the addition of the init function for backend samplers.

And llama_sampler_chain has also been updated to include the is_init flag,
and a info struct with a boolean to indicate if the sampler is a backend sampler:
```c++
struct llama_sampler_chain {
    llama_sampler_chain_params params;

    // has .backend_init() been called?
    bool is_init = false;

    struct info {
        bool is_backend;

        llama_sampler * ptr;
    };

    std::vector<info> samplers;

    // timing

    mutable int64_t t_sample_us;

    mutable int32_t n_sample;
};
```
So bascially all the CPU samplers are initialized first, and then if there exist
backend samplers support for them, and backend sampling is enbabled then a 
backend sampler will be initialized for that sampler and it will set on the
llama-context sampling_info instance:
```console
(gdb) p this.sampling
$16 = {samplers = std::map with 0 elements, logits = 0x0, logits_size = 0, sampled = 0x0, sampled_size = 0, probs = 0x0, 
  probs_size = 0, candidates = 0x0, candidates_size = 0, outputs_capacity = 0, logits_count = std::vector of length 0, capacity 0, 
  probs_count = std::vector of length 0, capacity 0, candidates_count = std::vector of length 0, capacity 0, 
  token_ids_full_vocab = std::vector of length 0, capacity 0}
(gdb) p this
$17 = (llama_context * const) 0x555556443ec0
```
And the CPU sampler chain will check this:
```c++
static void llama_sampler_chain_apply(struct llama_sampler * smpl, llama_token_data_array * cur_p) {
    auto * chain = (llama_sampler_chain *) smpl->ctx;

    time_meas tm(chain->t_sample_us, chain->params.no_perf);

    bool is_backend = chain->is_init;

    for (auto & smpl : chain->samplers) {
        /// smpl is actually of type llama_sampler_chain::info here
        if (is_backend && smpl.is_backend) {
            continue;
        }

        is_backend = false;

        if (smpl.ptr->iface->apply == nullptr) {
            continue;
        }

        llama_sampler_apply(smpl.ptr, cur_p);
    }
}
```
So if the sampler was initialized as a backend sampler we skip it here.

And simliarly for the backend apply function we only process the backend samplers:
```c++
static void llama_sampler_chain_backend_apply(
          struct llama_sampler      * smpl,
          struct ggml_context       * ctx,
          struct ggml_cgraph        * gf,
          struct llama_sampler_data * data) {
    auto * chain = (llama_sampler_chain *) smpl->ctx;

    GGML_ASSERT(chain->is_init && "llama_sampler_chain_backend_init() not called");

    for (auto & smpl : chain->samplers) {
        if (!smpl.is_backend) {
            break;
        }

        if (smpl.ptr->iface->backend_apply) {
            smpl.ptr->iface->backend_apply(smpl.ptr, ctx, gf, data);
        }
    }
}
```
In build_sampling there have also been a few changes:
```c++
    // add a dummy row of logits
    // this trick makes the graph static, regardless of which samplers are activated
    // this is important in order to minimize graph reallocations
    ggml_tensor * logits_t = ggml_pad(ctx0, res->t_logits, 0, 1, 0, 0);
```
Previously this was just using res->t_logits directly.
So differenent sequences can have different samplers configured. And all sequences
might not be active in every ubatch. We might have something like this:
```console
ubatch 1: sequences [0, 2, 5] are active
ubatch 2: sequences [1, 3]    are active
```
Without this padding the graph structure would change depending on what sequence
is currently being processed as t_logit shape would change, it would first
be [32000, 3] and then [32000, 2].
So with the padding we go from:
```
[32000, n_active_sequences]
```
To:
```
[32000, n_active_sequences + 1]
```
And we process all samplers regardless if that are active or not, active meaining
that the sequence they belong to is part of the current ubatch.
And the row idx is determined like this:
```c++
        // inactive samplers always work on the first row
        const auto row_idx = seq_to_logit_row.find(seq_id) != seq_to_logit_row.end() ? it->second : 0;
```
If the seq_id of the current sampler being iterated over is not in the current
ubatch then we use row 0 which is the dummy row we added.
So lets we have samplers for sequences [0, 1, 2] configured, but only sequences
[0, 2] are active:
```console
logits_t after padding:
row 0: [dummy data]   ← Sampler for seq_id=1 processes this (inactive)
row 1: [seq 0 logits] ← Sampler for seq_id=0 processes this (active)
row 2: [seq 2 logits] ← Sampler for seq_id=2 processes this (active)
```
The graph always has 3 sampler operations, but only the results for active
sequences are used.
