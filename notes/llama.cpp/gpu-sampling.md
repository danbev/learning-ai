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
- Top-k (see section below about an issue I ran into trying to implement this)
- Top-p
- Min-p
- additional?

### Top-k GPU sampling
I ran into an issue when trying to implement the top-k sampling on the GPU.
```c++
static void llama_sampler_gpu_top_k_apply_ggml(
        struct llama_sampler           * smpl,
        struct ggml_context            * ctx,
        struct ggml_cgraph             * gf,
        struct llama_sampler_ggml_data * ggml_data) {

    auto * ctx_data = (llama_sampler_gpu_top_k_ctx *) smpl->ctx;
    printf("gpu top-k: Building top-k sampler with k=%d\n", ctx_data->k);

    struct ggml_tensor * top_k = ggml_top_k(ctx, ggml_data->logits, ctx_data->k);
    ggml_set_name(top_k, "top_k");

    ggml_data->logits = ggml_get_rows(ctx, ggml_data->logits, top_k);
    ggml_build_forward_expand(gf, ggml_data->logits);
    ggml_data->size = ctx_data->k;
}
```
If we look at ggml_top_k we find that it is implemented like this:
```c++
// ggml_top_k

struct ggml_tensor * ggml_top_k(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        int                   k) {
    GGML_ASSERT(a->ne[0] >= k);

    struct ggml_tensor * result = ggml_argsort(ctx, a, GGML_SORT_ORDER_DESC);

    result = ggml_view_4d(ctx, result,
                k, result->ne[1], result->ne[2], result->ne[3],
                   result->nb[1], result->nb[2], result->nb[3],
                0);

    return result;
}
```
We can see that this is implemented using argsort:
```c++
struct ggml_tensor * ggml_argsort(
        struct ggml_context  * ctx,
        struct ggml_tensor   * a,
        enum ggml_sort_order   order) {
    GGML_ASSERT(a->ne[0] <= INT32_MAX);
    struct ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_I32, GGML_MAX_DIMS, a->ne);

    ggml_set_op_params_i32(result, 0, (int32_t) order);

    result->op     = GGML_OP_ARGSORT;
    result->src[0] = a;

    return result;
}
```
For the CUDA backend this is implemented using:
```c++
        case GGML_OP_ARGSORT:
            ggml_cuda_op_argsort(ctx, dst);
            break;
```
```c++
void ggml_cuda_op_argsort(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t ncols = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    enum ggml_sort_order order = (enum ggml_sort_order) dst->op_params[0];

    argsort_f32_i32_cuda(src0_d, (int *)dst_d, ncols, nrows, order, stream);
}
```
```c++
static void argsort_f32_i32_cuda(const float * x, int * dst, const int ncols, const int nrows, ggml_sort_order order, cudaStream_t stream) {
    // bitonic sort requires ncols to be power of 2
    const int ncols_pad = next_power_of_2(ncols);

    const dim3 block_dims(ncols_pad, 1, 1);
    const dim3 block_nums(1, nrows, 1);
    const size_t shared_mem = ncols_pad * sizeof(int);

    // FIXME: this limit could be raised by ~2-4x on Ampere or newer
    GGML_ASSERT(shared_mem <= ggml_cuda_info().devices[ggml_cuda_get_device()].smpb);

    if (order == GGML_SORT_ORDER_ASC) {
        k_argsort_f32_i32<GGML_SORT_ORDER_ASC><<<block_nums, block_dims, shared_mem, stream>>>(x, dst, ncols, ncols_pad);
    } else if (order == GGML_SORT_ORDER_DESC) {
        k_argsort_f32_i32<GGML_SORT_ORDER_DESC><<<block_nums, block_dims, shared_mem, stream>>>(x, dst, ncols, ncols_pad);
    } else {
        GGML_ABORT("fatal error");
    }
}
```
In the case that I'm testing the models vocabulary is 32000 tokens:
```console
(gdb) p *src0
$2 = {type = GGML_TYPE_F32, buffer = 0x555556989ce0, ne = {32000, 1, 1, 1}, nb = {4, 128000, 128000, 128000}, op = GGML_OP_NONE, 
  op_params = {0 <repeats 16 times>}, flags = 0, src = {0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0}, view_src = 0x0, 
  view_offs = 0, data = 0x7fff9ea20400, name = "leaf_0", '\000' <repeats 57 times>, extra = 0x0, 
  padding = "\000\000\000\000\000\000\000"}
```
But my devices shared memory per block is only 49152 bytes:
``` console
(gdb) p ggml_cuda_info().devices[ggml_cuda_get_device()].smpb
$9 = 49152
```
So perhaps for top-k sampling we might need a different algoritm that argsort
to avoid this shared memory limitation.

A similar limit can be found in the metal backend as well, in ggml-metal-device.m
there is the following check:
```c++
        case GGML_OP_ARGSORT:
            // TODO: Support arbitrary column width
            return op->src[0]->ne[0] <= 1024;
```
