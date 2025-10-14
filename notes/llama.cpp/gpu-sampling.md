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

One way could be to store the tensors in a struct like this:
```c++
    struct llama_sampler_ggml_data {
        struct ggml_tensor * ids;     // [n_vocab] - GGML_TYPE_I32
        struct ggml_tensor * logits;  // [n_vocab] - GGML_TYPE_F32
        struct ggml_tensor * probs;   // [n_vocab] - GGML_TYPE_F32
        int64_t size;                 // number of valid tokens in the arrays (<= n_vocab)
        int64_t selected;             // index in the array (-1 if not yet selected)
        bool sorted;                  // whether data is sorted by logits/probs
    };
};
```
The token ids can be stored as `I32` type tensors, and the logits and probabilities as `F32`.
Having separate tensors instead of perhaps packing data into a single tensor makes enables
easier operations on all of the logits or probabilities. Also if the types are different
this might also make sense to have them separate and not packed as well.

This would allow a function declaration to look something like this:
```c++
        void                   (*apply_ggml)(  struct llama_sampler * smpl,
                                                       ggml_context * ctx,
                                                        ggml_cgraph * gf,
                                            llama_sampler_ggml_data * ggml_data);
```
This way multiple GPU samplers can be chained together and they can all update
the graph with the operations they need to perform.

And `llama_sampler_chain` would then apply all the samplers calling `apply_ggml` for
each sampler in the chain, passing in a `ggml_cgraph` that is built up with all the
operations from each sampler.

While we want to avoid intermixing CPU and GPU samplers in the samplers chain, as this
would require converting and copying data between system memory to device memory, we should
support having GPU samplers at the start of the sampling chain. This way we can take
advantage of the logits already being on the GPU and perform some of the sampling
operations on the GPU before copying the data back to the CPU for any CPU samplers
to process later in the chain.

```c++
llama_token llama_sampler_sample(struct llama_sampler * smpl, struct llama_context * ctx, int32_t idx) {
    const auto * logits = llama_get_logits_ith(ctx, idx);

    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const int n_vocab = llama_vocab_n_tokens(vocab);
    ...
 
    // check the smpl chain and store the GPU samplers in a vector
    // Process the GPU samplers and afterwards create a llama_token_data_array
    // which can then be passed to the remaining CPU samplers in the chain.
}
```
Something like this perhaps. A suggestion following the above can be found in
[commit](https://github.com/danbev/llama.cpp/commit/c4d8e78d31e2ee5148c8b6bf8d564667a846b2c5).

### GPU Sampler parameters and state
Some samplers need to be able to accept parameters and also maintain state.
The tensors for the parameters and state need to be pre allocated made accessible
to the samplers.

If we take a top-k sampler as an example. This sampler needs to be initialized with
the 'k' value. This is possible in much the same way as the CPU implementation:
```c++
static struct llama_sampler * llama_sampler_gpu_init_top_k(int32_t k) {
    static const llama_sampler_i iface = {
        /*.name        =*/ llama_sampler_gpu_top_k_name,
        /*.accept      =*/ nullptr,
        /*.apply       =*/ nullptr,
        /*.reset       =*/ nullptr,
        /*.clone       =*/ nullptr,
        /*.free        =*/ llama_sampler_gpu_top_k_free,
        /*.apply_ggml  =*/ llama_sampler_gpu_top_k_apply_ggml
    };

    auto * ctx_data = new llama_sampler_gpu_top_k_ctx {
        /*.k =*/ k,
    };

    auto * sampler = new llama_sampler {
        /*.iface =*/ &iface,
        /*.ctx   =*/ ctx_data,
    };

    return sampler;
}
```
Currently, the GPU sampler are initialized before calling `llama_sampler_sample` so the
above works fine. In llama_sampler_sample is where we currently have the GPU sampler
processsing:
```c++
llama_token llama_sampler_sample(struct llama_sampler * smpl, struct llama_context * ctx, int32_t idx) {
    ...
        struct ggml_init_params params = {
            // TODO: need to take into account any tensors that GPU sampler may need.
            /*.mem_size   =*/ (ggml_tensor_overhead() * 5) + GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        struct ggml_context * ctx_sample = ggml_init(params);

        struct ggml_tensor * logits_t = ggml_new_tensor_1d(ctx_sample, GGML_TYPE_F32, n_vocab);
        struct ggml_tensor * ids      = ggml_new_tensor_1d(ctx_sample, GGML_TYPE_I32, n_vocab);
        struct ggml_tensor * probs    = ggml_new_tensor_1d(ctx_sample, GGML_TYPE_F32, n_vocab);
        struct ggml_tensor * selected = ggml_new_tensor_1d(ctx_sample, GGML_TYPE_I32, 1);

        // Select a GPU backend.
        // TODO: perhaps this should be configurable as to which GPU to use
        ggml_backend_t backend = nullptr;
        ggml_backend_buffer_type_t buft = nullptr;
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            auto * dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                backend = ggml_backend_dev_init(dev, nullptr);
                buft = ggml_backend_dev_buffer_type(dev);
                printf("Using GPU device '%s' for sampling\n", ggml_backend_dev_name(dev));
                break;
            }
        }
        ...

        struct ggml_cgraph * gf = ggml_new_graph(ctx_sample);

        struct llama_sampler_ggml_data ggml_data = {
            /*.ids      =*/ ids,
            /*.logits   =*/ logits_t,
            /*.probs    =*/ probs,
            /*.selected =*/ selected,
            /*.size     =*/ n_vocab,
            /*.sorted   =*/ false,
        };

        // Apply GPU samplers (add sampling operations to the graph)
        for (auto & smpl : gpu_samplers) {
            smpl.iface->apply_ggml(&smpl, ctx_sample, gf, &ggml_data);
        }

        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx_sample, buft);
        ...
}
```
A GPU sampler can create the tensors it needs in its apply_ggml function. But notice that the
graph is passed to this function, and we have to specify the memory size for the ggml_context
before this call happens.
So how do we know how much memory to allocate for the samplers tensors?  
Perhaps adding a callback for the gpu samplers be accaptable where the size of memory needed
by a sampler is returned? Something like:
```c++
        size_t                  (*size_ggml)(const  struct llama_sampler * smpl);
```
This could then be called when we gather the GPU samplers:
```c++
    std::vector<llama_sampler> gpu_samplers;
    size_t gpu_samplers_ggml_size = 0;
    if (smpl->iface->name && strcmp(smpl->iface->name(smpl), "chain") == 0) {
        for (int i = 0; i < llama_sampler_chain_n(smpl); i++) {
            auto * s = llama_sampler_chain_get(smpl, i);
            if (s->iface->apply_ggml) {
                gpu_samplers.push_back(*s);
                gpu_samplers_ggml_size += s->iface->size_ggml(s);

                // Remove the GPU sampler so that only CPU samplers remain in the chain
                llama_sampler_chain_remove(smpl, i);
            }
        }
```
We can then use this later when creating the context parameters:
```c++
        size_t total_ggml_size = gpu_samplers_ggml_size + (ggml_tensor_overhead() * 5) + GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
        printf("Total ggml size for GPU samplers: %zu bytes\n", total_ggml_size);
        struct ggml_init_params params = {
            // TODO: need to take into account any tensors that GPU sampler may need.
            /*.mem_size   =*/ total_ggml_size,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        struct ggml_context * ctx_sample = ggml_init(params);
```
This suggestion can be found in [commit](https://github.com/danbev/llama.cpp/commit/b0b2b904cc38bdafb07145d034a336c211af1537).


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
I've updated the suggestion above and we can check the samplers in the chain
and if they are all GPU samplers then we can avoid creating the
llama_token_data_array altogether.

* Originally I have specified that either GPU samplers or CPU samplers would be
  used in a sampling chain. But there was a suggestion to allow mixing them in the
  sense that the GPU samplers would be at the start of the chain and CPU samplers
  after them.

* It was also brought up that CPU samplers may require addition tensors for storing
  parameters and state. These tensors need to be preallocated and made availabe to the
  GPU samplers in some way. An example of such a sampler is the dist sampler that needs
  to store the RNG (random number generator) state.

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
