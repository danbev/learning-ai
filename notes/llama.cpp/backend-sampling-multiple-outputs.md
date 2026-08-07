## Backend sampling multiple outputs support

### clamp issue in backend dist sampler
The dist sampler uses cumulative sum to pick a token.

Given probabilities: [0.1, 0.2, 0.3, 0.4] for 4 candidates
Step 1: Compute cumsum
```console
  probs:  [0.1,  0.2,  0.3,  0.4]
  cumsum: [0.1,  0.3,  0.6,  1.0]
```
Step 2: Generate random value (let's say uniform = 0.55)
Step 3: Compare cumsum with uniform
```console
diff = cumsum - uniform
       = [  0.1,   0.3,  0.6, 1.0] - 0.55
       = [-0.45, -0.25, 0.05, 0.45]
```
Step 4: Apply step function (1 if > 0, else 0)
```console
 mask = step(diff)
       = [0, 0, 1, 1]
```
Step 5: Sum the mask
```console
  idxf = sum(mask) = 2
```
Step 6: Calculate index
```console
idx = (-1 * idxf) + n
      = (-1 * 2) + 4
      = 2  ✓ (selects the 3rd candidate, index 2)
```
Now, with the 4 candidates, but uniform = 1.5:

Step 3: Compare
```console
  diff = cumsum - uniform
       = [0.1, 0.3, 0.6, 1.0] - 1.5
       = [-1.4, -1.2, -0.9, -0.5]  // ALL NEGATIVE!
```
Step 4: Apply step
```console
  mask = step(diff)
       = [0, 0, 0, 0]  // ALL ZEROS!
```
Step 5: Sum
```console
  idxf = sum(mask) = 0
```
Step 6: Calculate index WITHOUT clamp
```console
  idx = (-1 * 0) + 4
      = 4  out of bounds! (valid indices are 0-3)
```

The Fix with Clamp
```c++
  idxf = ggml_clamp(ctx, idxf, 1.0f, mask->ne[0]);
```
This forces idxf to be in range [1, n]:

After clamp:
```c++
  idxf = clamp(0, 1.0, 4.0) = 1  // Clamped to minimum of 1
```
Calculate index WITH clamp:
```console
  idx = (-1 * 1) + 4
      = 3  ✓ (selects the last candidate, which makes sense)
```

Visual Summary
```console
  Without clamp:          With clamp:
  n = 4 candidates        n = 4 candidates
  idxf = 0                idxf = 1 (clamped)
  idx = 0 + 4 = 4         idx = -1 + 4 = 3 ✓

  Valid indices: 0-3      Valid indices: 0-3
  Trying to access: 4     Trying to access: 3
  Crash!                  Success!
```

### PR review notes (https://github.com/ggml-org/llama.cpp/pull/2)
I think the best way to understand this is to look at one of the added test
cases which is `test_backend_multi_sequence_multi_output_dist`.

This will have two sequences which both have backend samplers configured. The
samplers will be a temp sampler, and a dist sampler. They will have the same
configuration and the only difference will be the seed they use.
```c++
static void test_backend_multi_sequence_multi_output_dist(const test_params & params) {
    ...
    const uint32_t seeds[] = { 88, 1337 };
    const float temp = 10.0f;

    // Two backend sampler chains
    llama_sampler_ptr chain_0(llama_sampler_chain_init(llama_sampler_chain_default_params()));
    llama_sampler_chain_add(chain_0.get(), llama_sampler_init_temp(temp));
    llama_sampler_chain_add(chain_0.get(), llama_sampler_init_dist(seeds[0]));

    llama_sampler_ptr chain_1(llama_sampler_chain_init(llama_sampler_chain_default_params()));
    llama_sampler_chain_add(chain_1.get(), llama_sampler_init_temp(temp));
    llama_sampler_chain_add(chain_1.get(), llama_sampler_init_dist(seeds[1]));

    std::vector<llama_sampler_seq_config> configs = {
        { 0, chain_0.get() },
        { 1, chain_1.get() },
    };
```
These sampler configs will be used with the text_context:
```c++
                                        n_max_seq n_ubatch 
                                           |     +----+
                                           ↓     ↓
    test_context test_ctx(params, configs, 2, 4, 0, 2);
                                              ↑     ↑
                                              |     +-------- n_sampling_outputs_per_seq_max 
                                              n_output_max
```
The test also creates a reference test context which does not have any backend
samplers:
```console
    std::vector<llama_sampler_seq_config> reference_configs;
    test_context reference_ctx(params, reference_configs, 2, 4);
```
The test then creates two sequences, each one having two tokens, the bos token
and the eos token (notice the opposite order):
```c++
    const llama_token seq_tokens[2][2] = {
        { llama_vocab_bos(vocab), llama_vocab_eos(vocab) },
        { llama_vocab_eos(vocab), llama_vocab_bos(vocab) },
    };
```
```console
(gdb) p seq_tokens
$17 = {{151643, 151645}, {151645, 151643}}
        (bos)    (eos)     (eos)   (bos)

(gdb) p llama_vocab_bos(vocab)
$18 = 151643

(gdb) p llama_vocab_eos(vocab)
$19 = 151645
```

Next we create a new llama_batch for the sequences, setting n_tokens to 4, and
each sequence 
```c++
    llama_batch batch = llama_batch_init(4, 0, 1);
```
And then it populates the tokens in the batch using:
```c++
    for (int pos = 0; pos < 2; ++pos) {
        common_batch_add(batch, seq_tokens[0][pos], pos, { 0 }, true);
        common_batch_add(batch, seq_tokens[1][pos], pos, { 1 }, true);
    }
```
The first row in the for loop is adding the token for first position and associating
it with the sequence 0 ({0}), and all tokens should have logits generated for them.
And similar for the second sequence.

The test context is then decode:
```c++
    GGML_ASSERT(llama_decode(test_ctx.ctx.get(), batch) == 0);
```
We need to take a look at llama_sampler_dist as it has been updated:
```c++
struct llama_sampler_dist : public llama_sampler_backend {
    const uint32_t seed;
          uint32_t seed_cur;

    std::mt19937 rng;

    // multi-output backend draws are committed as an accepted prefix
    bool backend_transactional;
    std::mt19937 rng_backend;
    size_t n_backend_draws_generated;
    size_t n_backend_draws_committed;

    // inputs for the current sampling graph
    std::vector<ggml_tensor *> inp_uniforms;
};
```
Here rng_backend is for the speculative sampling which advances, remember that
RGN is stateful, but tokens might need to be rejected and therefor rng is used
for the main sampling and only advances when the token is accepted. As we will
see below rng_backend is create by copying the current rng.

In llama_decode we have the following before the process function is called:
```c++
    // start a new sampling transaction for this logical batch
    for (const auto & entry : sampling.samplers) {
        llama_sampler_backend_begin(entry.second);
    }
```
```c++
void llama_sampler_backend_begin(llama_sampler * sampler) {
    GGML_ASSERT(sampler != nullptr);

    if (sampler->iface == &llama_sampler_chain_i) {
        auto * chain = (llama_sampler_chain *) sampler->ctx;
        for (auto & entry : chain->samplers) {
            if (!entry.is_backend) {
                break;
            }
            llama_sampler_backend_begin(entry.ptr);
        }
    } else if (sampler->iface == &llama_sampler_dist_i) {
        auto * ctx = (llama_sampler_dist *) sampler->ctx;
        if (ctx->backend_transactional) {
            ctx->rng_backend = ctx->rng;
            ctx->n_backend_draws_generated = 0;
            ctx->n_backend_draws_committed = 0;
        }
    }
}
```
Notice that this is performing a full copy of ctx->rng to ctx->rng_backend so
they are completely independent from this point onward. So generating a random
number from rng_backend will not affect the rng. But also keep in mind that they
have the exact same state at this point in time, so if we only generated one
single random number from both of them they would be the exact same.

The backend_set_input function is later called from the process_ubatch:
```c++
static void llama_sampler_dist_backend_set_input(struct llama_sampler * smpl) {
    auto * sctx = (llama_sampler_dist *) smpl->ctx;

    GGML_ASSERT(!sctx->inp_uniforms.empty());

    // We sample in double precision and cast to float to match rnd numbers of
    // llama_dampler_dist which uses double precision (sampling from
    // std::uniform_real_distribution<double> and
    // std::uniform_real_distribution<float> with same rng will produce
    // different sequences).
    std::uniform_real_distribution<double> dist(0.0f, 1.0f);

    auto & rng = sctx->backend_transactional ? sctx->rng_backend : sctx->rng;

    for (auto * inp_uniform : sctx->inp_uniforms) {
        GGML_ASSERT(inp_uniform != nullptr);

        const float rnd = dist(rng);
        ggml_backend_tensor_set(inp_uniform, &rnd, 0, sizeof(float));

        if (sctx->backend_transactional) {
            ++sctx->n_backend_draws_generated;
        }
    }
}
```
First a mutable reference to a generator is obtained, either the backend one
or the main one. Then a distribution is created.
Notice that we don't just have one uniform tensor anymore, but instead a vector
of them. First we generate a random number using the generator, this advances it
and we write it to the input tensor. And notice that this is casting it down from
double to float.
And the number of backend draws generated is incremented for keeping track of
how many.


The test also creates a reference random number generator and a distribution
similar to what the dist sampler will do internally:
```console
    std::mt19937 reference_rngs[] = {
        std::mt19937(seeds[0]),
        std::mt19937(seeds[1]),
    };
    std::uniform_real_distribution<double> reference_dist(0.0, 1.0);
```

After that the test will iterate over all the tokens in the batch and will
process the sequence id of each token:
```c++
    for (int i = 0; i < batch.n_tokens; ++i) {
        const llama_seq_id seq_id = batch.seq_id[i][0];

        llama_sampler * chain = seq_id == 0 ? chain_0.get() : chain_1.get();
```
This gets the sampler chain for the sequence, either chain 0 or 1.
Next the sampler will be sampled:
```c++
        const llama_token backend_token = llama_sampler_sample(chain, test_ctx.ctx.get(), i);
```
The llama_sampler_sample call will end up in:
```c++
llama_token llama_sampler_sample(struct llama_sampler * smpl, struct llama_context * ctx, int32_t idx) {
    const llama_token   sampled_token  = llama_get_sampled_token_ith     (ctx, idx);
    const float *       sampled_probs  = llama_get_sampled_probs_ith     (ctx, idx);
    const float *       sampled_logits = llama_get_sampled_logits_ith    (ctx, idx);
    const llama_token * sampled_ids    = llama_get_sampled_candidates_ith(ctx, idx);

    // If a backend sampler has already sampled a token, return it.
    if (sampled_token != LLAMA_TOKEN_NULL) {
        LLAMA_LOG_DEBUG("%s: Backend sampler selected token for idx %d. Skipping CPU samplers\n", __func__, idx);
        llama_sampler_accept(smpl, sampled_token);
        return sampled_token;
    }
```
In our case the backend sampler will have already sampled a token so this will
call llama_sampler_accept and then return the token id.

The accept function looks like this:
```c++
static void llama_sampler_dist_accept(struct llama_sampler * smpl, llama_token token) {
    GGML_UNUSED(token);

    auto * sctx = (llama_sampler_dist *) smpl->ctx;

    if (!sctx->backend_transactional ||
            sctx->n_backend_draws_committed >= sctx->n_backend_draws_generated) {
        return;
    }

    std::uniform_real_distribution<double> dist(0.0f, 1.0f);
    dist(sctx->rng);
    ++sctx->n_backend_draws_committed;
}
```
Notice that this only advances the main rng by sampling from the distribution
but does not actually use the returned value because it does not need to, the
backend sampler has already accepted the token and we don't need to pick anything,
just advance the state.

There has been a change to llama-graph.h:
```console
diff --git a/src/llama-graph.h b/src/llama-graph.h
index 160e29413..675dece84 100644
--- a/src/llama-graph.h
+++ b/src/llama-graph.h
@@ -883,10 +883,10 @@ public:
 
     std::vector<ggml_tensor *> t_layer_inp;
 
-    std::map<llama_seq_id, ggml_tensor *> t_sampled_logits;
-    std::map<llama_seq_id, ggml_tensor *> t_candidates;
-    std::map<llama_seq_id, ggml_tensor *> t_sampled;
-    std::map<llama_seq_id, ggml_tensor *> t_sampled_probs;
+    std::vector<ggml_tensor *> t_sampled;
+    std::vector<ggml_tensor *> t_sampled_probs;
+    std::vector<ggml_tensor *> t_sampled_logits;
+    std::vector<ggml_tensor *> t_candidates;
 
     std::vector<llm_graph_input_ptr> inputs;
     std::vector<llm_graph_fused_node> fused_nodes;
```
So there are now just simple vectors and not maps.

There is also the following addition in build_sampling:
```console
    for (const auto & entry : samplers) {
        if (entry.second->iface->backend_reset) {
            entry.second->iface->backend_reset(entry.second);
        }
    }
```
This is what the backend_reset looks like for the dist backend sampler:
```console
static void llama_sampler_dist_backend_reset(struct llama_sampler * smpl) {
    auto * sctx = (llama_sampler_dist *) smpl->ctx;
    sctx->inp_uniforms.clear();
}
```
This is clearning the uniform input vector for all the samplers (2 in our case):
```console
(gdb) ptype sctx->inp_uniforms
type = std::vector<ggml_tensor*>
```

Next we create a mapping of sequence ids to output row indices:
```c++
    std::map<llama_seq_id, std::vector<uint32_t>> sampling_rows;
    uint32_t n_rows = 0;
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        if (ubatch.output[i]) {
            sampling_rows[ubatch.seq_id[i][0]].push_back(n_rows++);
        }
    }
```
Notice that we only increment n_rows when the token in the ubatch is set as
output (true) so we are only counting tokens that produce logit outputs.
```console
i=0: seq_id=0, output=true → sampling_rows[0].push_back(0), n_rows=1
i=1: seq_id=1, output=true → sampling_rows[1].push_back(1), n_rows=2
i=2: seq_id=0, output=true → sampling_rows[0].push_back(2), n_rows=3
i=3: seq_id=1, output=true → sampling_rows[1].push_back(3), n_rows=4

(gdb) p sampling_rows
$57 = std::map with 2 elements = {
[0] = std::vector of length 2, capacity 2 = {0, 2},
[1] = std::vector of length 2, capacity 2 = {1, 3}}
```
So sampling_rows[0] = {0, 2} means that sequence 0's two output set a logit rows 0
and 2. And likewise sampling_rows[1] = {1, 3} means that sequence 1's two outputs
are set to logits rows 1 and 3.


Next we have:
```c++
    static const std::vector<uint32_t> dummy_row = { 0 };

    for (const auto & [seq_id, sampler] : samplers) {
        const auto it = sampling_rows.find(seq_id);
```
And our sampling_rows map looks like this at this point:
```console
(gdb) p sampling_rows
$54 = std::map with 2 elements = {
[0] = std::vector of length 2, capacity 2 = {0, 2}, 
[1] = std::vector of length 2, capacity 2 = {1, 3}}
```

Then we iterate over all the rows, the outputs for this sequence. And notice
that this will use rows[i] to offset into the logits_t tensor when creating the
view of the logits for this sequence, so that will be the start of the row in
memory:
```c++
    for (uint32_t i = 0; i < rows.size(); ++i) {
        ggml_tensor * logits_seq = ggml_view_1d(ctx0, logits_t, logits_t->ne[0], rows[i] * logits_t->nb[1]);
        ggml_format_name(logits_seq, "logits_seq_%d_%u", seq_id, i);
```
Then we create the sampler data and call backend_apply:
```console
            struct llama_sampler_data data = {
                /*.logits       =*/ logits_seq,
                /*.probs        =*/ nullptr,
                /*.sampled      =*/ nullptr,
                /*.candidates   =*/ nullptr,
            };

            assert(sampler->iface->backend_apply);
            sampler->iface->backend_apply(sampler, ctx0, gf, &data);
```
In our case the temp sampler will be called first. And this will work on the
logits view that we set above. This will build up the computation graph for this
backend sampler.
And then the dist sampler will be called to do the same. In dist apply we have:
```c++
    ggml_tensor * inp_uniform = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_format_name(inp_uniform, "uniform_%zu", sctx->inp_uniforms.size());
    ggml_set_input(inp_uniform);
    sctx->inp_uniforms.push_back(inp_uniform);
```
Notice that we are storing the inp_uniform tensors here. These will later be
used in backend_set_inputs when generating the uniform input values.

Just to set back a little here. In build_sampling we iterated over all the
active samplers and each output row the have, 2 in our case. So backend_apply
will be called 4 times for the complete batch. These will all be added to the
same computation graph. Each one will have its own inp_uniform tensor which is
the input that we set. So sequence 0 will have 2 and sequence 1 will have 2.
These are separate samplers so they have separate sampler context (their own
counters). As we saw before there is a new API function that is named backend_reset
which resets this vector.
```c++ 
static void llama_sampler_dist_backend_set_input(struct llama_sampler * smpl) {
    auto * sctx = (llama_sampler_dist *) smpl->ctx;

    GGML_ASSERT(!sctx->inp_uniforms.empty());

    std::uniform_real_distribution<double> dist(0.0f, 1.0f);

    auto & rng = sctx->backend_transactional ? sctx->rng_backend : sctx->rng;

    for (auto * inp_uniform : sctx->inp_uniforms) {
        GGML_ASSERT(inp_uniform != nullptr);

        const float rnd = dist(rng);
        ggml_backend_tensor_set(inp_uniform, &rnd, 0, sizeof(float));

        if (sctx->backend_transactional) {
            ++sctx->n_backend_draws_generated;
        }
    }
}
```

We have to keep in mind that backend_set_input is called for every ubatch
but backend_apply is only called when the graph if rebuilt. When the graph is
reused backend_apply is not run and only backend_set_inputs runs again to create
new random numbers and copy them into the existing tensors. So we can't just
clear the list after backend_set_inputs is done as when the graph is reused
inp_uniforms would be empty and the assert would fire. Instead what the PR does
is that it calls backend_reset in build_sampling, which is what is called when
the graph need to be rebuilt.

So when the graph is rebuilt we know that build_sampling is called. And when
the graph is reused then backend_set_input will be called which needs to retain
its ggml_tensor pointers. When the graph is rebuilt it will have a new ggml_context
and perhaps we could store a pointer to it just to detect the last known context,
and if this changes then we would clear the inp_uniforms and set the last_ctx
to the new one and so on.


After the apply functions have been called our llama_sampler_data will look like this:
```console
(gdb) p data
$63 = {logits = 0x5555556f5bf0, probs = 0x5555556f6040, sampled = 0x5555556f6a50, candidates = 0x0}
```
We then procees any sampled data:
```console
            if (data.sampled != nullptr) {
                if (active) {
                    res->t_sampled[rows[i]] = data.sampled;
                }
                outs[1] = data.sampled;
                ggml_build_forward_select(gf, outs.data(), outs.size(), i_out);
            }
```
```console
(gdb) p rows[i]
$64 = 0
(gdb) p res->t_sampled
$65 = std::vector of length 4, capacity 4 = {0x0, 0x0, 0x0, 0x0}
```
Since this sampler is active this will then set t_sampled[0] to the data.sampled.
And it sets outs[1] to this as well so that it can be forward expanded (we need
an array of outs if the sampler is inactive and only one will be selected, but
this keep the graph static).
And we do the same thing for:
```console
            if (data.probs != nullptr) {
                if (active) {
                    res->t_sampled_probs[rows[i]] = data.probs;
                }
                outs[1] = data.probs;
                ggml_build_forward_select(gf, outs.data(), outs.size(), i_out);
            }

            if (data.logits != nullptr) {
                if (active) {
                    res->t_sampled_logits[rows[i]] = data.logits;
                }
                outs[1] = data.logits;
                ggml_build_forward_select(gf, outs.data(), outs.size(), i_out);
            }

            if (data.candidates != nullptr) {
                if (active) {
                    res->t_candidates[rows[i]] = data.candidates;
                }
                outs[1] = data.candidates;
                ggml_build_forward_select(gf, outs.data(), outs.size(), i_out);
            }
```
So the same thing will happend for the next output (row) and then for the second
sequence (sampler).
So at this point we have build the graph for the samplers. This will return
back to llama_context::process_ubatch.
```console
        gf = model.build_graph(gparams);
```
A bit further down we will call:
```console
        res->set_inputs(&ubatch);
```
And this will call the dist samplers backend_set_input that we saw above which
will create a distribution and then iterate over the uniform tensors.

This PR has a nice solution for mapping batch token slots to output rows/slots.
What I mean is that my initial solution was to in build_sampling create a mapping
from sequence_id to the output row. But the new code in this PR simply used 
vectors and they map to the batch token slot so there is not need for the whole
mapping. And this also means that in decode the tensor copies can be simlified
to:
```c++
// async copy the sampling data from the backend to the host
copy_tensor_async_rows(res->t_sampled, sampling.sampled, 1, n_outputs_prev, sched.get());
copy_tensor_async_rows(res->t_sampled_logits, sampling.logits,stride, n_outputs_prev, sched.get(), &sampling.logits_count);
copy_tensor_async_rows(res->t_sampled_probs, sampling.probs, stride, n_outputs_prev, sched.get(), &sampling.probs_count);
copy_tensor_async_rows(res->t_candidates, sampling.candidates, stride, n_outputs_prev, sched.get(), &sampling.candidates_count);
```
So notice that this is passing a vector of tensors:
```console
(gdb) p res->t_sampled
$3 = std::vector of length 4, capacity 4 = {0x5555556f6a50, 0x5555556f8cd0, 0x5555556f7b90, 0x5555556f9e10}
(gdb) ptype res->t_sampled
type = std::vector<ggml_tensor*>
```
So for token ubatch.token[0] the sampled token for it is in res->t_sampled.

And sampling is of type sampling_info:
```c++
    struct sampling_info {
        // !samplers.empty() to check if any samplers are active
        std::map<llama_seq_id, llama_sampler *> samplers;

        buffer_view<float>       logits     = {nullptr, 0};
        buffer_view<llama_token> sampled    = {nullptr, 0};
        buffer_view<float>       probs      = {nullptr, 0};
        buffer_view<llama_token> candidates = {nullptr, 0};

        std::vector<uint32_t> logits_count;
        std::vector<uint32_t> probs_count;
        std::vector<uint32_t> candidates_count;

        // optimization
        std::vector<llama_token> token_ids_full_vocab;
    };

    sampling_info sampling;
```
And buffer_view is just a struct that has a data type as the pointer to the
data and a size:
```c++
template <typename T>
struct buffer_view {
    T * data;
    size_t size = 0;

    bool has_data() const {
        return data && size > 0;
    }
};
```
And these host buffers are initialized in output_reserve in llama-context.cpp:
```c++
    if (has_sampling) {
        sampling.logits = {(float *) (base + offset), (size_t)(n_vocab*n_outputs_max)};
        offset += sampling.logits.size * sizeof(float);
        ...
    }
```
Notice that this is a single pointer but if we have multiple outputs, for example
2 outputs then n_outputs_max will be 2 and we will have space for both outputs
in this buffer.

So lets look at the inputs to the first call:
```c++
copy_tensor_async_rows(res->t_sampled, sampling.sampled, 1, n_outputs_prev, sched.get());
```
Notice that this call uses a stride of 1 which is becaue each sampled token is
a single llama_token integer so they are just one slot apart. Recall that we
saw that we have a single buffer view and this case we have n_outputs=4.

And we have n_outputs_prev which is because a batch can be split into multiple
micro batches (ubatch) and we need to write each output into the correct position
of the host buffer. This is just a counter which increments for each ubatch.

So stride controls how far apart consecutive rows are in the destination buffer,
1 element for a single token, n_vocab elements for a full logit/prob distribution.
And row_offset ensures that if the batch was split into ubatches, each ubatch's
outputs land after the previous ones in the flat host buffer rather than
overwriting from the start.

```c++
template<typename T>
static void copy_tensor_async_rows(
    const std::vector<ggml_tensor *> & tensors,
    const buffer_view<T> & dst,
    size_t stride,
    uint32_t row_offset,
    ggml_backend_sched_t sched,
    std::vector<uint32_t> * counts = nullptr) {
    if (!dst.has_data()) {
        return;
    }

    for (size_t i = 0; i < tensors.size(); ++i) {
        auto * tensor = tensors[i];
        if (tensor == nullptr) {
            continue;
        }

        const uint32_t row = row_offset + i;
        const size_t n_elements = ggml_nelements(tensor);
        GGML_ASSERT(ggml_is_contiguous(tensor) && "sampling tensor must be contiguous for async copy");
        GGML_ASSERT(n_elements <= stride);
        GGML_ASSERT((size_t) row * stride + n_elements <= dst.size);

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, tensor);
        T * row_ptr = dst.data + (size_t) row * stride;
        ggml_backend_tensor_get_async(backend, tensor, row_ptr, 0, ggml_nbytes(tensor));

        if (counts) {
            GGML_ASSERT(row < counts->size());
            (*counts)[row] = n_elements;
        }
    }
}
```
Notice that if the tensor is nullptr then the loop continues, and this means
that counts is not updated, it will remain as 0 which is the initalized value
in output_reserve.


One last part that was been updated in this pr is graph_reserve:
```c++

```

### backend init
```c++
static bool llama_sampler_chain_backend_init(
        struct llama_sampler       * smpl,
        ggml_backend_buffer_type_t   buft,
        uint32_t                     n_outputs_per_seq_max) {
    auto * chain = (llama_sampler_chain *) smpl->ctx;

    GGML_ASSERT(chain->is_init == false && "llama_sampler_chain_backend_init() called twice");

    chain->is_init = true;

    bool res = true;
    bool backend_prefix = true;

    for (auto & smpl : chain->samplers) {
        bool res_cur = backend_prefix;

        // to be able to run a sampler on the backend, it has to:
        // - have the .backend_init() API implemented
        // - return true during .backend_init()
        // - support the requested per-sequence output limit
        if (res_cur && smpl.ptr->iface->backend_init) {
            if (!smpl.ptr->iface->backend_init(smpl.ptr, buft, n_outputs_per_seq_max)) {
                res_cur = false;
            }
        } else {
            res_cur = false;
        }

        smpl.is_backend = res_cur;
        backend_prefix = res_cur;

        res = res && res_cur;
    }

    auto probe = llama_sampler_backend_probe_graph(smpl, 1024*1024, GGML_DEFAULT_GRAPH_SIZE, false);
    chain->n_nodes = llama_sampler_backend_probe_n_nodes(probe);

    return res;
}
```
So a chain of sampler can be thought of as:
```console
chain: [temp] [top-k] [penalties] [dist]
       |            | |                |
       +------------+ +----------------+
       backend prefix  cpu suffix
```
backend prefix is a running boolean meaning: "are we still inside the prefix".
Is starts as true and once it goes to false it never goes back. So a
backend-CPU-backend chain is not possible.

llama_context::graph_max_nodes:
```c++
    uint32_t n_sampling_nodes = 0;
    uint32_t n_sampling_nodes_max = 0;
    for (const auto & [seq_id, sampler] : sampling.samplers) {
        const uint32_t n_nodes = llama_sampler_backend_n_nodes(sampler);
        n_sampling_nodes += n_nodes;

        if (cparams.n_sampling_outputs_per_seq_max > 1) {
            n_sampling_nodes_max = std::max(n_sampling_nodes_max, n_nodes);
        }
    }
```
This is counting all the nodes, actually tensors for all backend samplers. And
if there are more than one output per sequence then it it updates
n_sampling_nodes_max to store the largest number of tensors.
