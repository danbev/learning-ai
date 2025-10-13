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

We want to avoid intermixing CPU and GPU samplers as this would require converting
and copying data from system memory to device memory. So we should use either
only GPU samplers, or only CPU samplers in the chain. We could add a check for
to see if the samplers in the chain are GPU only and then avoid creating the
llama_token_data_array and only create the ggml_data struct in llamaa_sampler_sample.
```c++
llama_token llama_sampler_sample(struct llama_sampler * smpl, struct llama_context * ctx, int32_t idx) {
    const auto * logits = llama_get_logits_ith(ctx, idx);

    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const int n_vocab = llama_vocab_n_tokens(vocab);
    ...

    bool all_gpu = ...
    
    if (all_gpu) {
       ... 
    } else {
        ...
    }
}
```
Something like this perhaps.

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
