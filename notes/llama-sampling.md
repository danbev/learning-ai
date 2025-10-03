### Llama.cpp Sampling API
This document is about sampling in llama.cpp. I've written about sampling
strategies in [sampling.md](sampling.md) and these notes are specifically about
the implementation in llama.cpp.

### Interface
The interface for the sampling API is mainly defined in llama.h:

There is a `llama_sampler` that looks like this:
```c++
    struct llama_sampler {
        struct llama_sampler_i  * iface;
        llama_sampler_context_t   ctx;
    };
```
Where the llama_sample_context_t is defined as:
```c++
    typedef void * llama_sampler_context_t;
```
So this is a void pointer meaning that this context can be anything that the
sampler implementation needs.

The `llama_sampler_i` is the interface for the sampler.
```c++
struct llama_sampler_i {
   const char* (*name) (const struct llama_sampler* smpl); // can be NULL
   void (*accept) (struct llama_sampler* smpl, llama_token token); // can be NULL
   void (*apply)  (struct llama_sampler* smpl, llama_token_data_array * cur_p); // required
   void (*reset)  (struct llama_sampler* smpl); // can be NULL
   struct llama_sampler* (*clone) (const struct llama_sampler* smpl); // can be NULL if ctx is NULL
   void (*free) (struct llama_sampler* smpl);  // can be NULL if ctx is NULL
};
```
So that is the interface for an implementation of a sampler. The way we use
a sampler is by calling functions in llama.h that take struct llama_sampler
pointer as an argument:
```c++
    LLAMA_API const char* llama_sampler_name(const struct llama_sampler* smpl);
    LLAMA_API void llama_sampler_accept(struct llama_sampler* smpl, llama_token token);
    LLAMA_API void llama_sampler_apply (struct llama_sampler* smpl, llama_token_data_array * cur_p);
    LLAMA_API void llama_sampler_reset (struct llama_sampler* smpl);
    LLAMA_API struct llama_sampler* llama_sampler_clone (const struct llama_sampler * smpl);
    LLAMA_API void llama_sampler_free(struct llama_sampler* smpl);
```

The following samplers are provided by default and declared in llama.h:
```c++
    LLAMA_API struct llama_sampler * llama_sampler_init_greedy     (void);
    LLAMA_API struct llama_sampler * llama_sampler_init_dist       (uint32_t seed);
    LLAMA_API struct llama_sampler * llama_sampler_init_softmax    (void);
    LLAMA_API struct llama_sampler * llama_sampler_init_top_k      (int32_t k);
    LLAMA_API struct llama_sampler * llama_sampler_init_top_p      (float   p, size_t min_keep);
    LLAMA_API struct llama_sampler * llama_sampler_init_min_p      (float   p, size_t min_keep);
    LLAMA_API struct llama_sampler * llama_sampler_init_tail_free  (float   z, size_t min_keep);
    LLAMA_API struct llama_sampler * llama_sampler_init_typical    (float   p, size_t min_keep);
    LLAMA_API struct llama_sampler * llama_sampler_init_temp       (float   t);
    LLAMA_API struct llama_sampler * llama_sampler_init_temp_ext   (float   t, float   delta, float exponent);

    LLAMA_API struct llama_sampler * llama_sampler_init_mirostat(
                             int32_t   n_vocab,
                            uint32_t   seed,
                               float   tau,
                               float   eta,
                             int32_t   m);

    LLAMA_API struct llama_sampler * llama_sampler_init_mirostat_v2(
```

Lets take look at an example:
```c++
    struct llama_sampler* sampler = llama_sampler_init_greedy();
```

In llama.cpp/src/llama-sampling.cpp we have the implementation:
```c++
struct llama_sampler * llama_sampler_init_greedy() {
    return new llama_sampler {
        /* .iface = */ &llama_sampler_greedy_i,
        /* .ctx   = */ nullptr,
    };
}

static struct llama_sampler_i llama_sampler_greedy_i = {
    /* .name   = */ llama_sampler_greedy_name,
    /* .accept = */ nullptr,
    /* .apply  = */ llama_sampler_greedy_apply,
    /* .reset  = */ nullptr,
    /* .clone  = */ nullptr,
    /* .free   = */ nullptr,
};

static const char * llama_sampler_greedy_name(const struct llama_sampler * /*smpl*/) {
    return "greedy";
}

static void llama_sampler_greedy_apply(struct llama_sampler * /*smpl*/, llama_token_data_array * cur_p) {
    cur_p->selected = 0;
    for (size_t i = 1; i < cur_p->size; ++i) {
        if (cur_p->data[i].logit > cur_p->data[cur_p->selected].logit) {
            cur_p->selected = i;
        }
    }
}
```
Now, lets say that we call `llama_decode` with a prompt and then call:
```c++
        const llama_token new_token_id = llama_sampler_sample(sampler, ctx, -1);
```
Lets take a closer look at the `llama_sampler_sample` function:
```c++
llama_token llama_sampler_sample(struct llama_sampler * smpl, struct llama_context * ctx, int32_t idx) {
    const auto * logits = llama_get_logits_ith(ctx, idx);

    const int n_vocab = llama_n_vocab(llama_get_model(ctx));

    // TODO: do not allocate each time
    std::vector<llama_token_data> cur(n_vocab);
    for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
        cur[token_id] = llama_token_data{token_id, logits[token_id], 0.0f};
    }
```
So we can see that we are getting the logits from the context and using -1 as
the index. Then we get the size of the vocabulary and create a vector of
`llama_token_data` structs with that size. In this case the vocabulary size is
```console
(gdb) p cur.size()
$29 = 32000
```
Notice that this is actually creating 32000 `llama_token_data` structs using
the default initialization. Then the for loop is iterating over the vocabulary
and overwriting the entries with new instances of `llama_token_data` structs.

Perhaps this could be rewritten to use `emplace_back` instead of overwriting:
```c++
std::vector<llama_token_data> cur;
cur.reserve(n_vocab);

for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
    cur.emplace_back(token_id, logits[token_id], 0.0f);
}
```

And `llama_token_data` is defined as:
```console
(gdb) ptype llama_token_data
type = struct llama_token_data {
    llama_token id;
    float logit;
    float p;
}
```
After an instance of `llama_token_data_array` will be created using the
vector created above:
```c++
    llama_token_data_array cur_p = {
        /* .data       = */ cur.data(),
        /* .size       = */ cur.size(),
        /* .selected   = */ -1,
        /* .sorted     = */ false,
    };

    typedef struct llama_token_data_array {
        llama_token_data * data;
        size_t size;
        int64_t selected; // this is the index in the data array (i.e. not the token id)
        bool sorted;
    } llama_token_data_array;
```
And this `llama_token_data_array` will be passed to the `llama_sampler_apply`:
```c++
    llama_sampler_apply(smpl, &cur_p);
```
And the `llama_sampler_apply` simply delegates to the interfaces apply
implementation which is the greedy sampler in this case:
```c++
void llama_sampler_apply(struct llama_sampler * smpl, struct llama_token_data_array * cur_p) {
    GGML_ASSERT(smpl->iface->apply);
    smpl->iface->apply(smpl, cur_p);
}
```
And the the data array looks like this at this point:
```console
(gdb) p *cur_p
$36 = {data = 0x555555f201c0, size = 32000, selected = -1, sorted = false}
```

```c++
static void llama_sampler_greedy_apply(struct llama_sampler* smpl, llama_token_data_array* cur_p) {
    cur_p->selected = 0;
    for (size_t i = 1; i < cur_p->size; ++i) {
        if (cur_p->data[i].logit > cur_p->data[cur_p->selected].logit) {
            cur_p->selected = i;
        }
    }
}
```
So first selected it set to 0 from being -1 and then we iterate over the 32000
`llama_token_data` structs which recall have a `logit`, a `id`, and a `p`
(probability) field.
The if statement is checking the current logit and if it has a higher value
than the current selected logit (currently 0) then selected is updated to the
index of that element.
So the logit with the highest value is selected. This will then return us to
`llama_sampler_sample`:
```c++

    auto token = cur_p.data[cur_p.selected].id;
```
```console
(gdb) p token
$48 = 13

(gdb) p ctx.model.vocab.id_to_token[13]
$49 = {text = "<0x0A>", score = 0, attr = LLAMA_TOKEN_ATTR_BYTE}
```
This is the `LF` token (hex 0A).

And then `llama_sampler_accepts` will be called with the selected token id:
```c++
    llama_sampler_accept(smpl, token);
```
This will also delegate to the samplers implementation of the accept function:
```c++
void llama_sampler_accept(struct llama_sampler * smpl, llama_token token) {
    if (smpl->iface->accept) {
        smpl->iface->accept(smpl, token);
    }
}
```
In the case of the greedy sampler this is a accept is not implemented:
```console
(gdb) p *smpl.iface
$53 = {name = 0x555555797838 <llama_sampler_greedy_name(llama_sampler const*)>,
accept = 0x0, apply = 0x55555579784d <llama_sampler_greedy_apply(llama_sampler*,
llama_token_data_array*)>,
reset = 0x0, clone = 0x0, free = 0x0}
```
The last thing that happens in `llama_sampler_sample` is that the token id is
returned.


### Order of samplers
The default samplers that will be used for genreation are the following:
```console
    penalties;dry;top_n_sigma;top_k;typ_p;top_p;min_p;xtc;temperature)
```
These come from `arg.cpp`:
```c++
    add_opt(common_arg(
        {"--samplers"}, "SAMPLERS",
        string_format("samplers that will be used for generation in the order, separated by \';\'\n(default: %s)", sampler_type_names.c_str()),
        [](common_params & params, const std::string & value) {
            const auto sampler_names = string_split<std::string>(value, ';');
            params.sampling.samplers = common_sampler_types_from_names(sampler_names, true);
        }
```
And saqmpler_type_names is created at the start of the function:
```c++
    std::string sampler_type_chars;
    std::string sampler_type_names;
    for (const auto & sampler : params.sampling.samplers) {
        sampler_type_chars += common_sampler_type_to_chr(sampler);
        sampler_type_names += common_sampler_type_to_str(sampler) + ";";
    }
    sampler_type_names.pop_back();
```
```console
(gdb) set print array on
(gdb) p params.sampling.samplers
$4 = std::vector of length 9, capacity 9 = {
  COMMON_SAMPLER_TYPE_PENALTIES,
  COMMON_SAMPLER_TYPE_DRY,
  COMMON_SAMPLER_TYPE_TOP_N_SIGMA,
  COMMON_SAMPLER_TYPE_TOP_K,
  COMMON_SAMPLER_TYPE_TYPICAL_P,
  COMMON_SAMPLER_TYPE_TOP_P,
  COMMON_SAMPLER_TYPE_MIN_P,
  COMMON_SAMPLER_TYPE_XTC,
  COMMON_SAMPLER_TYPE_TEMPERATURE
}
```
One thing to note is that there is also a bias sampler that is not included in
the above list, but is added in common/sampling.cpp:
```c++
struct common_sampler * common_sampler_init(const struct llama_model * model, const struct common_params_sampling & params) {
    const llama_vocab * vocab = llama_model_get_vocab(model);

    ...
    llama_sampler_chain_add(result->chain,
            llama_sampler_init_logit_bias(
                llama_vocab_n_tokens(vocab),
                params.logit_bias.size(),
                params.logit_bias.data()));
```
After this the rest of the samplers are added in the order specified:
```c++
    if (params.mirostat == 0) {
        for (const auto & cnstr : params.samplers) {
            switch (cnstr) {
            ...
                case COMMON_SAMPLER_TYPE_PENALTIES:
                    llama_sampler_chain_add(result->chain,
                        llama_sampler_init_penalties(params.penalty_last_n, params.penalty_repeat, params.penalty_freq, params.penalty_present));
                    break;
                default:
                    GGML_ASSERT(false && "unknown sampler type");
            }
        }
        llama_sampler_chain_add(result->chain, llama_sampler_init_dist(params.seed));
```
And notice that the distribution sampler is added at the end of the chain which
is what samples from the probabilities that that the sampler chains outputs.
```console
(gdb) p ((llama_sampler_chain *)result.chain.ctx).samplers[0]
$10 = (llama_sampler *) 0x55555593dab0
(gdb) p *((llama_sampler_chain *)result.chain.ctx).samplers[0]
$11 = {iface = 0x7ffff7f6d280 <llama_sampler_logit_bias_i>, ctx = 0x555555dfe210}

(gdb) p *((llama_sampler_chain *)result.chain.ctx).samplers[1]
$12 = {iface = 0x7ffff7f6d1c0 <llama_sampler_penalties_i>, ctx = 0x555555dfd6a0}

(gdb) p *((llama_sampler_chain *)result.chain.ctx).samplers[2]
$13 = {iface = 0x7ffff7f6d240 <llama_sampler_dry_i>, ctx = 0x5555559614e0}

(gdb) p *((llama_sampler_chain *)result.chain.ctx).samplers[3]
$14 = {iface = 0x7ffff7f6d200 <llama_sampler_top_n_sigma_i>, ctx = 0x55555593df30}

(gdb) p *((llama_sampler_chain *)result.chain.ctx).samplers[4]
$15 = {iface = 0x7ffff7f6cf40 <llama_sampler_top_k_i>, ctx = 0x555555930ec0}

(gdb) p *((llama_sampler_chain *)result.chain.ctx).samplers[5]
$16 = {iface = 0x7ffff7f6d000 <llama_sampler_typical_i>, ctx = 0x555555930960}

(gdb) p *((llama_sampler_chain *)result.chain.ctx).samplers[6]
$17 = {iface = 0x7ffff7f6cf80 <llama_sampler_top_p_i>, ctx = 0x555555e006b0}

(gdb) p *((llama_sampler_chain *)result.chain.ctx).samplers[7]
$18 = {iface = 0x7ffff7f6cfc0 <llama_sampler_min_p_i>, ctx = 0x555555930fa0}

(gdb) p *((llama_sampler_chain *)result.chain.ctx).samplers[8]
$19 = {iface = 0x7ffff7f6d0c0 <llama_sampler_xtc_i>, ctx = 0x555555e02ff0}

(gdb) p *((llama_sampler_chain *)result.chain.ctx).samplers[9]
$20 = {iface = 0x7ffff7f6d080 <llama_sampler_temp_ext_i>, ctx = 0x55555593cf50}

(gdb) p *((llama_sampler_chain *)result.chain.ctx).samplers[10]
$21 = {iface = 0x7ffff7f6cf00 <llama_sampler_dist_i>, ctx = 0x555555e043a0}
```
```
llama_token common_sampler_sample(struct common_sampler * gsmpl, struct llama_context * ctx, int idx, bool grammar_first) {
    gsmpl->set_logits(ctx, idx);

    auto & grmr  = gsmpl->grmr;
    auto & chain = gsmpl->chain;
    auto & cur_p = gsmpl->cur_p; // initialized by set_logits
```
`set_logits` does the following:
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
};
```
So we can see here that the vector for each token id a `llama_token_data` struct
is created and it is initilized with the token id and the logits from the model
and the probability is initialized to 0.

Notice that initially the index into `cur` is the exact same as the token id and
one might wonder why we need to also store the token id in the struct. The reason
for this is that some sampler will sort the array based on the logits or other
wise reorder the array. So the index in the array is not necessarily the same as
the token id after samplers have been applied..

The sampling started by calling:
```c++
llama_token common_sampler_sample(struct common_sampler * gsmpl, struct llama_context * ctx, int idx, bool grammar_first) {
    gsmpl->set_logits(ctx, idx);

    auto & grmr  = gsmpl->grmr;
    auto & chain = gsmpl->chain;
    auto & cur_p = gsmpl->cur_p; // initialized by set_logits

    if (grammar_first) {
        llama_sampler_apply(grmr, &cur_p);
    }

    llama_sampler_apply(chain, &cur_p);
```
```c++
void llama_sampler_apply(struct llama_sampler * smpl, struct llama_token_data_array * cur_p) {
    GGML_ASSERT(smpl->iface->apply);
    smpl->iface->apply(smpl, cur_p);
}
```

And this delegates to `llama_sampler_chain_apply`:
```c++
static void llama_sampler_chain_apply(struct llama_sampler * smpl, llama_token_data_array * cur_p) {
    auto * chain = (llama_sampler_chain *) smpl->ctx;

    time_meas tm(chain->t_sample_us, chain->params.no_perf);

    for (auto * smpl : chain->samplers) {
        llama_sampler_apply(smpl, cur_p);
    }
}
```
And this is the same function as we saw before but this time the sampler is not
the chain sampler but a concrete sampler, in this case the logit bias sampler which
is the first in the chain of samplers:
```c++
void llama_sampler_apply(struct llama_sampler * smpl, struct llama_token_data_array * cur_p) {
    GGML_ASSERT(smpl->iface->apply);
    smpl->iface->apply(smpl, cur_p);
}
```

### logit bias sampler
```c++
static void llama_sampler_logit_bias_apply(struct llama_sampler * smpl, llama_token_data_array * cur_p) {
    auto * ctx = (llama_sampler_logit_bias *) smpl->ctx;

    if (ctx->logit_bias.empty()) {
        return;
    }

    ctx->to_search.clear();

    // update the candidates that have not been shuffled in the vocabulary (i.e. idx == id)
    for (const auto & lb : ctx->logit_bias) {
        if (lb.token >= 0 && cur_p->size > (size_t) lb.token && cur_p->data[lb.token].id == lb.token) {
            cur_p->data[lb.token].logit += lb.bias;
        } else {
            ctx->to_search.push_back(lb);
        }
    }

    if (ctx->to_search.empty()) {
        return;
    }

    // search for the remaining candidates that were not found in the previous step
    for (size_t i = 0; i < cur_p->size; ++i) {
        for (const auto & lb : ctx->to_search) {
            if (cur_p->data[i].id == lb.token) {
                cur_p->data[i].logit += lb.bias;
                break;
            }
        }
    }
}
```
The first look is a fast path where the underlying vector of logits has not
been reordered by any previous samplers (so the index is the same as the token
id). The general idea of this sampler is that it allows us to modifiy specific
digits in the vocabulary by adding a bias to the logit for that token. We can
add a positive value to increase the probability of a specific token, and also
specify a large negative number to effectively remove a token from being sampled.

Note, that this sampler operates on logits and does not use the probability field

### penalities sampler
This sampler implements several techniques to reduce repetition in generated
text. This sampler applies penalties to tokens that have already appeared in
recent generation history (the `penalty_last_n` tokens).

```c++
struct llama_sampler_penalties {
    const int32_t penalty_last_n;
    const float   penalty_repeat;
    const float   penalty_freq;
    const float   penalty_present;

    ring_buffer<llama_token> prev;

    // a frequency map to count token occurrences
    std::unordered_map<llama_token, int> token_count;
};
```

```c++
static void llama_sampler_penalties_apply(struct llama_sampler * smpl, llama_token_data_array * cur_p) {
    auto * ctx = (llama_sampler_penalties *) smpl->ctx;

    if ((ctx->penalty_last_n == 0) ||
        (ctx->penalty_repeat == 1.0f && ctx->penalty_freq == 0.0f && ctx->penalty_present == 0.0f)) {
        return;
    }

    // Apply frequency and presence penalties to the cur_p
    for (size_t i = 0; i < cur_p->size; ++i) {
        const auto token_iter = ctx->token_count.find(cur_p->data[i].id);
        if (token_iter == ctx->token_count.end()) {
            continue;
        }

        const int count = token_iter->second;

        assert(count > 0 && count <= ctx->penalty_last_n);

        // The academic publication that described this technique actually just only divided, but that would cause tokens with negative logits to become more likely, which is obviously wrong.
        // This is common fix for this problem, which is to multiply by the penalty instead of dividing.
        if (cur_p->data[i].logit <= 0) {
            cur_p->data[i].logit *= ctx->penalty_repeat;
        } else {
            cur_p->data[i].logit /= ctx->penalty_repeat;
        }

        cur_p->data[i].logit -= float(count) * ctx->penalty_freq + float(count > 0) * ctx->penalty_present;
    }

    cur_p->sorted = false;
}
```
So these can all be applied at the same time. I though they might only be used
one by one. But the repeat penalty comes first which may modify the logits. Then
we multiply the count by the requency penalty, and also multiply 0 or 1 depending
if the count is greater or equal to zero with the penalty_present (boolean value 
so we either apply by the penalty of we don't).

Note, that this sampler operates on logits and does not use the probability field

### llama_sampler_dry
TODO:

### llama_sample_top_n_sigma
```c++
struct llama_sampler_top_n_sigma {
    const float n;
};
```
Instead of keeping a fixed number of top tokens (like top-k) or a probability
mass (like top-p), this sampler uses statistical distance from the maximum logit
```c++
static void llama_sampler_top_n_sigma_apply(struct llama_sampler * smpl, llama_token_data_array * cur_p) {
    auto * ctx = (llama_sampler_top_n_sigma *) smpl->ctx;

    if (ctx->n <= 0.0f || cur_p->size <= 1) {
        return;
    }

    // find max logit
    float max = cur_p->data[0].logit;
    float logits_sum = 0;
    size_t valid_count = 0;
    for (size_t i = 0; i < cur_p->size; ++i) {
        // Only count non-negative infinity values
        if (cur_p->data[i].logit != -INFINITY) {
            if (cur_p->data[i].logit > max) {
                max = cur_p->data[i].logit;
            }
            logits_sum += cur_p->data[i].logit;
            valid_count++;
        }
    }
    // calculate mean
    float mean = valid_count > 0 ? logits_sum/valid_count : 0;

    // calculate standard deviation
    float acc = 0;
    for (size_t i = 0; i < cur_p->size; ++i) {
        // Skip -infinity in std calculation
        if (cur_p->data[i].logit != -INFINITY) {
            acc += pow(cur_p->data[i].logit - mean, 2);
        }
    }
    float std = valid_count > 0 ? sqrt(acc/valid_count) : 0;

    // apply mask
    for (size_t i = 0; i < cur_p->size; ++i) {
        if (cur_p->data[i].logit < max - (ctx->n * std)) {
            cur_p->data[i].logit = -INFINITY;
        }
    }

    llama_sampler_softmax_impl(cur_p, true);
}
```

Note: if enabled this sampler operates on the logits, but it will also call
`llama_sampler_softmax_impl` which will compute the probabilities from the logits
and update the probability field in the `llama_token_data` structs.

### llama_sampler_top_k
The sampler keeps only the `k` tokens with the highest logits and discards the
rest.
```c++
struct llama_sampler_top_k {
    const int32_t k;
};
```

```c++
static void llama_sampler_top_k_apply(struct llama_sampler * smpl, llama_token_data_array * cur_p) {
    auto * ctx = (llama_sampler_top_k *) smpl->ctx;
    llama_sampler_top_k_impl(cur_p, ctx->k);
}

static void llama_sampler_top_k_impl(llama_token_data_array * cur_p, int32_t k) {
    if (k <= 0) {
        return;
    }

    k = std::min(k, (int) cur_p->size);

    // Sort scores in descending order
    if (!cur_p->sorted) {
        llama_token_data_array_partial_sort_inplace(cur_p, k);
    }

    // Also sets cur_p->size to k, overriding the value that was potentially
    // set by llama_token_array_partial_sort_inplace. Should this perhaps really
    // be done in an else clause?
    cur_p->size = k;
}

static void llama_token_data_array_partial_sort_inplace(llama_token_data_array * cur_p, int npartial) {
    // Notice that this comparator is using the logits!
    static const auto comp = [](const llama_token_data & a, const llama_token_data & b) {
        return a.logit > b.logit;
    };

    if (npartial <= 128) {
        std::partial_sort(cur_p->data, cur_p->data + npartial, cur_p->data + cur_p->size, comp);

        cur_p->size = npartial;
        cur_p->sorted = true;

        return;
    }

    std::vector<llama_token_data> tmp;

    llama_token_data_array_partial_sort(*cur_p, npartial, tmp);

    std::copy(tmp.data(), tmp.data() + npartial, cur_p->data);

    // Setting the size to npartial means that the rest of the array is ignored.
    cur_p->size = npartial;
    cur_p->sorted = true;
}
```
```console
(gdb) p ctx->k
$44 = 40
(gdb) p *cur_p
$46 = {data = 0x5555585b3c80, size = 262144, selected = -1, sorted = false}
```
So after this sampler has processed the token data array the size will be set
to 40:
```console
(gdb) p *cur_p
$51 = {data = 0x5555585b3c80, size = 40, selected = -1, sorted = true}   
```
So after this sampler, if k=40, then the only logits that the rest of the
samplers can process are these 40 token data entries (which contain both logits
and probabilities remember).

### llama_sampler_typical
This sampler works on the probabilities and not the logits. So it first
converts the logits to probabilities using softmax.
```c++
struct llama_sampler_typical {
    const float  p;
    const size_t min_keep;
};
```
```c++
static void llama_sampler_typical_apply(struct llama_sampler * smpl, llama_token_data_array * cur_p) {
    auto * ctx = (llama_sampler_typical *) smpl->ctx;

    // Reference implementation:
    // https://github.com/huggingface/transformers/compare/main...cimeister:typical-sampling:typical-pr
    if (ctx->p >= 1.0f) {
        return;
    }

    // Compute the softmax of logits and calculate entropy
    llama_sampler_softmax_impl(cur_p, true);

    float entropy = 0.0f;
    for (size_t i = 0; i < cur_p->size; ++i) {
        entropy += -cur_p->data[i].p * logf(cur_p->data[i].p);
    }

    // Compute the absolute difference between negative log probability and entropy for each candidate
    std::vector<float> shifted_scores;
    for (size_t i = 0; i < cur_p->size; ++i) {
        float shifted_score = fabsf(-logf(cur_p->data[i].p) - entropy);
        shifted_scores.push_back(shifted_score);
    }

    // Sort tokens based on the shifted_scores and their corresponding indices
    std::vector<size_t> indices(cur_p->size);
    std::iota(indices.begin(), indices.end(), 0);

    std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
        return shifted_scores[a] < shifted_scores[b];
    });

    // Compute the cumulative probabilities
    float cum_sum = 0.0f;
    size_t last_idx = indices.size();

    for (size_t i = 0; i < indices.size(); ++i) {
        size_t idx = indices[i];
        cum_sum += cur_p->data[idx].p;

        // Check if the running sum is greater than typical or if we have kept at least min_keep tokens
        if (cum_sum > ctx->p && (ctx->min_keep == 0 || i >= ctx->min_keep - 1)) {
            last_idx = i + 1;
            break;
        }
    }

    // Resize the output vector to keep only the locally typical tokens
    std::vector<llama_token_data> cur_p_new;
    for (size_t i = 0; i < last_idx; ++i) {
        size_t idx = indices[i];
        cur_p_new.push_back(cur_p->data[idx]);
    }

    // Replace the data in cur_p with the cur_p_new data
    std::copy(cur_p_new.begin(), cur_p_new.end(), cur_p->data);
    cur_p->size = cur_p_new.size();
    cur_p->sorted = false;
}
```
So after this sampler the probabilites for the logits in the token data array
have been modified and the size of the array may have been reduced.

So lets say we have the following state of the token data array:
```console
(gdb) p *cur_p
$19 = {data = 0x5555585b3c80, size = 262144, selected = -1, sorted = false}
(gdb) p cur_p->data[0]
$25 = {id = 0, logit = -14.8716631, p = 0}
```
Then `top_k` is applied with k=40:
```console
(gdb) p *cur_p
$20 = {data = 0x5555585b3c80, size = 40, selected = -1, sorted = true}
(gdb) p cur_p->data[0]
$31 = {id = 108, logit = 19.8492393, p = 0}
```
Then later `top_p` is applied (which calls soft max:
```console
(gdb) p *cur_p
$34 = {data = 0x5555585b3c80, size = 27, selected = -1, sorted = true}
(gdb) p cur_p->data[0]
$35 = {id = 108, logit = 19.8492393, p = 0.272734016}
```
So top_p first computes the softmax and then resized the token data array but
the probabilites will not add up to 1.0 anymore because the size was reduced.
```(gdb) p cur_p->data[0]
$35 = {id = 108, logit = 19.8492393, p = 0.272734016}
(gdb) p cur_p->data[1]
$36 = {id = 563, logit = 18.9221611, p = 0.107923076}
(gdb) p cur_p->data[2]
$37 = {id = 4733, logit = 18.6403351, p = 0.0814177021}
(gdb) p cur_p->data[3]
$38 = {id = 564, logit = 18.4178543, p = 0.0651773438}
(gdb) p cur_p->data[4]
$39 = {id = 623, logit = 18.2506371, p = 0.0551410653}
(gdb) p cur_p->data[5]
$40 = {id = 19565, logit = 18.2467232, p = 0.054925669}
(gdb) p cur_p->data[6]
$41 = {id = 107, logit = 18.0632076, p = 0.0457167737}
(gdb) p cur_p->data[7]
$42 = {id = 669, logit = 17.8008919, p = 0.0351684578}
(gdb) p cur_p->data[8]
$43 = {id = 691, logit = 17.6138248, p = 0.0291682985}
(gdb) p cur_p->data[9]
$44 = {id = 753, logit = 17.4331284, p = 0.0243464503}
(gdb) p cur_p->data[10]
$45 = {id = 1174, logit = 17.1942959, p = 0.0191739686}
(gdb) p cur_p->data[11]
$46 = {id = 236743, logit = 17.1441193, p = 0.0182356201}
(gdb) p cur_p->data[12]
$47 = {id = 496, logit = 17.1277504, p = 0.0179395545}
(gdb) p cur_p->data[13]
$48 = {id = 506, logit = 17.0165386, p = 0.0160514023}
(gdb) p cur_p->data[14]
$49 = {id = 1030, logit = 16.9550114, p = 0.0150935724}
(gdb) p cur_p->data[15]
$50 = {id = 562, logit = 16.8741608, p = 0.0139212767}
(gdb) p cur_p->data[16]
$51 = {id = 568, logit = 16.6988392, p = 0.011682556}
(gdb) p cur_p->data[17]
$52 = {id = 2375, logit = 16.6446133, p = 0.0110659283}
(gdb) p cur_p->data[18]
$53 = {id = 138, logit = 16.3903847, p = 0.00858178828}
(gdb) p cur_p->data[19]
$54 = {id = 255999, logit = 16.2614384, p = 0.00754357362}
(gdb) p cur_p->data[20]
$55 = {id = 799, logit = 16.1067486, p = 0.00646243524}
(gdb) p cur_p->data[21]
$56 = {id = 109, logit = 16.08395, p = 0.00631676801}
(gdb) p cur_p->data[22]
$57 = {id = 2981, logit = 16.0823326, p = 0.00630655931}
(gdb) p cur_p->data[23]
$58 = {id = 815, logit = 16.0728855, p = 0.00624726154}
(gdb) p cur_p->data[24]
$59 = {id = 668, logit = 16.0606232, p = 0.00617112266}
(gdb) p cur_p->data[25]
$60 = {id = 672, logit = 16.021904, p = 0.00593674881}
(gdb) p cur_p->data[26]
$61 = {id = 625, logit = 15.9493284, p = 0.00552114891}
(gdb) p cur_p->data[27]
$62 = {id = 1176, logit = 15.8668432, p = 0.0050840131}
```

After all samplers have been applied we get:
```console
(gdb) p *cur_p
$65 = {data = 0x5555585b3c80, size = 16, selected = 3, sorted = true}
```


### llama_sampler_top_p
```c++
struct llama_sampler_top_p {
    const float  p;
    const size_t min_keep;

    std::vector<llama_token_data> buf_sort;
};
```
```c++
static void llama_sampler_top_p_apply(struct llama_sampler * smpl, llama_token_data_array * cur_p) {
    auto * ctx = (llama_sampler_top_p *) smpl->ctx;

    if (ctx->p >= 1.0f) {
        return;
    }

    llama_sampler_softmax_impl(cur_p, false);

    size_t k = cur_p->size;
    auto * pdata = cur_p->data;

    auto & buf_sort = ctx->buf_sort;

    // if not sorted, try adaptive top-k sorting
    if (!cur_p->sorted && cur_p->size > 1024) {
        k = std::min<size_t>(256, cur_p->size);
        llama_token_data_array_partial_sort(*cur_p, k, buf_sort);
        pdata = buf_sort.data();
    } else if (!cur_p->sorted) {
        // small candidates -> sort inplace
        llama_token_data_array_partial_sort_inplace(cur_p, k);
    }

    // Compute the cumulative probabilities
    float cum_sum = 0.0f;
    size_t last_idx = cur_p->size;

    for (size_t i = 0; i < cur_p->size; ++i) {
        cum_sum += pdata[i].p;

        // Check if the running sum is at least p or if we have kept at least min_keep tokens
        // we set the last index to i+1 to indicate that the current iterate should be included in the set
        if (cum_sum >= ctx->p && i + 1 >= ctx->min_keep) {
            last_idx = i + 1;
            break;
        }

        // we exceeded the current top-k heuristic -> increase k and continue
        if (!cur_p->sorted && i == k - 1) {
            k = cur_p->size;
            llama_token_data_array_partial_sort(*cur_p, k, buf_sort);
            pdata = buf_sort.data();
        }
    }

    // Resize the output vector to keep only the top-p tokens
    if (!cur_p->sorted) {
        std::copy(buf_sort.data(), buf_sort.data() + last_idx, cur_p->data);
        cur_p->sorted = true;
    }

    cur_p->size = last_idx;
}
```
Notice that this sampler changes the size of the token data array. So after this
sampler the size may be smaller than before, for example in my debugging session:
```console
(gdb) p last_idx
$14 = 27
```
Now, at this point the previous sampler calculated the probabilities from using
the logits that current at that point. If a sampler in the chain was to use
the probabilites of the tokens in the current token data array they would not
be valid, they no longer sum to 1. 

### llama_sampler_dist
This is the final sampler in the chain and it samples from the probabilites.

While working on an isssue (see #simplify-llama-token-data below) I ran into
an issue that I though I had somehow introduced but it turns out it the same
behavior is present in the master branch as well.


```console
sampler seed: 1680199764
sampler params:
	repeat_last_n = 64, repeat_penalty = 1.000, frequency_penalty = 0.000,
    presence_penalty = 0.000 dry_multiplier = 0.000, dry_base = 1.750,
    dry_allowed_length = 2, dry_penalty_last_n = 4096 top_k = 40, top_p = 0.950,
    min_p = 0.050, xtc_probability = 0.000, xtc_threshold = 0.100,
    typical_p = 1.000, top_n_sigma = -1.000, temp = 0.800 mirostat = 0,
    mirostat_lr = 0.100, mirostat_ent = 5.000

sampler chain: logits -> logit-bias -> penalties -> dry -> top-n-sigma
               -> top-k -> typical -> top-p -> min-p -> xtc -> temp-ext -> dist
````

```c++
static void llama_sampler_dist_apply(struct llama_sampler * smpl, llama_token_data_array * cur_p) {
    auto * ctx = (llama_sampler_dist *) smpl->ctx;

    ...
    float max_l = cur_p->data[0].logit;
    if (!cur_p->sorted) {
        for (size_t i = 1; i < cur_p->size; ++i) {
            max_l = std::max(max_l, cur_p->data[i].logit);
        }
    }

    // apply softmax to obtain the probabilities
    double sum_cum = 0.0f;
    for (size_t i = 0; i < cur_p->size; ++i) {
        float p = expf(cur_p->data[i].logit - max_l);
        cur_p->data[i].p = p;
        sum_cum += p;
    }
```
The above is setting the `max_l` value to the first logit which is the highest
logit as the array is sorted. Then the exponential of each logit minus the max
logit is calculated which will be used as the denominator in the softmax
calculation later by summing them into `sum_cum`.

And at this point `cur_p` looks like this:
```console
(gdb) p *cur_p
$1 = {data = 0x5555585b3c80, size = 16, selected = 0, sorted = true}
(gdb) p cur_p->data[0]
$2 = {id = 108, logit = 24.8115482, p = 1}
(gdb) p cur_p->data[1]
$3 = {id = 563, logit = 23.6527004, p = 0.313847572}
(gdb) p cur_p->data[2]
$4 = {id = 4733, logit = 23.3004189, p = 0.220660627}
(gdb) p cur_p->data[3]
$5 = {id = 564, logit = 23.0223179, p = 0.167088717}
(gdb) p cur_p->data[4]
$6 = {id = 623, logit = 22.8132954, p = 0.135571942}
(gdb) p cur_p->data[5]
$7 = {id = 19565, logit = 22.808403, p = 0.1349103}
(gdb) p cur_p->data[6]
$8 = {id = 107, logit = 22.57901, p = 0.107255846}
(gdb) p cur_p->data[7]
$9 = {id = 669, logit = 22.2511139, p = 0.0772711709}
(gdb) p cur_p->data[8]
$10 = {id = 691, logit = 22.0172806, p = 0.061159648}
(gdb) p cur_p->data[9]
$11 = {id = 753, logit = 21.7914104, p = 0.0487944931}
(gdb) p cur_p->data[10]
$12 = {id = 1174, logit = 21.4928703, p = 0.0362006612}
(gdb) p cur_p->data[11]
$13 = {id = 236743, logit = 21.4301491, p = 0.0339998491}
(gdb) p cur_p->data[12]
$14 = {id = 496, logit = 21.409687, p = 0.0333112143}
(gdb) p cur_p->data[13]
$15 = {id = 506, logit = 21.2706738, p = 0.0289879665}
(gdb) p cur_p->data[14]
$16 = {id = 1030, logit = 21.1937637, p = 0.0268420801}
(gdb) p cur_p->data[15]
$17 = {id = 562, logit = 21.092701, p = 0.0242619198}
```
Next, we have:
```c++
    std::uniform_real_distribution<double> dist(0.0f, 1.0f);
    const double rnd = dist(ctx->rng);

          double sum_run = 0.0f;
    const double sum_tgt = sum_cum*rnd;

    bool found = false;
    for (size_t i = 0; i < cur_p->size; ++i) {
        if (!found) {
            // accumulate probs until we reach the target sum
            sum_run += cur_p->data[i].p;
            if (sum_run >= sum_tgt) {
                cur_p->selected = i;
                found = true;
            }
        }

        // normalize probs
        cur_p->data[i].p /= sum_cum;
    }
```
So the above will iterate over all the tokens in the token data array. And this
is adding the exponential values (not the probabilities) to the running sum,
and then checking if the running sum is greater than the target sum. For the
first value we have:
```console
(gdb) p cur_p->data[i].p
$20 = 1
(gdb) p sum_run
$21 = 0
(gdb) n
651	            if (sum_run >= sum_tgt) {
(gdb) p sum_run
$22 = 1
```

### Simplify `llama_token_data`
In the above code we saw the definition of `llama_token_data`:
```c++
    // TODO: simplify (https://github.com/ggml-org/llama.cpp/pull/9294#pullrequestreview-2286561979)
    typedef struct llama_token_data {
        llama_token id; // token id
        float logit;    // log-odds of the token
        float p;        // probability of the token
    } llama_token_data;
```
Notice that this has the token id which is the index in the vocabulary, the logit
which is the raw score from the model. Now, the probability field is something
that needs to be computed from the logits, for example using softmax. But not
all sampling strategies need or use the probability field. For example top-k
samples where the k highest logits are selected and the rest are filtered out.

This struct is included in the following array:
```c++
    typedef struct llama_token_data_array {
        // TODO: consider SoA
        // NOTE: this pointer can be modified by the samplers
        llama_token_data * data;
        size_t size;
        int64_t selected; // this is the index in the data array (i.e. not the token id)
        bool sorted;      // note: do not assume the data is sorted - always check this flag
    } llama_token_data_array;
```

After making the changes to only have a single score field instead of logit and
p field I added the following assert to `llama_sampler_temp_impl` and a few
other samplers which operate on the raw logits:
```console
static void llama_sampler_temp_impl(llama_token_data_array * cur_p, float temp) {
    GGML_ASSERT(cur_p->raw);
    if (temp <= 0.0f) {
        // find the token with the highest logit and set the rest to -inf
        size_t max_i = 0;
        float  max_l = cur_p->data[0].score;

        for (size_t i = 1; i < cur_p->size; ++i) {
            if (cur_p->data[i    ].score > max_l) {
                cur_p->data[max_i].score = -INFINITY;
                max_i = i;
                max_l = cur_p->data[i].score;
            } else {
                cur_p->data[i].score = -INFINITY;
            }
        }

        return;
    }

    for (size_t i = 0; i < cur_p->size; ++i) {
        cur_p->data[i].score /= temp;
    }
}
```
But this caused a test in `test-thread-safety` to fail with:
```console
/home/danbev/work/ai/llama.cpp/src/llama-sampling.cpp:263: GGML_ASSERT(cur_p->raw) failed
```
The samplers configured in the test are as follows:
```console
(gdb) p params.samplers
$7 = std::vector of length 9, capacity 9 =
{
COMMON_SAMPLER_TYPE_PENALTIES,
COMMON_SAMPLER_TYPE_DRY,
COMMON_SAMPLER_TYPE_TOP_N_SIGMA,
COMMON_SAMPLER_TYPE_TOP_K,
COMMON_SAMPLER_TYPE_TYPICAL_P,
COMMON_SAMPLER_TYPE_TOP_P,
COMMON_SAMPLER_TYPE_MIN_P,
COMMON_SAMPLER_TYPE_XTC,
COMMON_SAMPLER_TYPE_TEMPERATURE}
```
If these are applied in this order, that is they are chained, then temperature
will come last and it will expect the score to be raw logits. But there other
samplers like TOP_P will have converted the logits into probabilities. So this
has introduced a ordering dependency between the samplers which is not ideal but
is something that was there previously as well only that there was not an error
but it just worked as there were separate fields for the logits and the
probabilites.
This is how the test is started:
```console
$ gdb --args build/bin/test-thread-safety "-hf" "ggml-org/models" "-hff" "tinyllamas/stories15M-q4_0.gguf" "-ngl" "99" "-p" "The meaning of life is" "-n" "128" "-c" "256" "-ub" "32" "-np" "4" "-t" "2"
```
So there is nothing specific about the sampling parameters that have been set
here. If we inspect the samplers right after the the  struct has been created:
```c++
int main(int argc, char ** argv) {
    common_params params;
```
struct common_params {
```console
(gdb) p params.sampling.samplers
$11 = std::vector of length 9, capacity 9 = {COMMON_SAMPLER_TYPE_PENALTIES, COMMON_SAMPLER_TYPE_DRY, 
  COMMON_SAMPLER_TYPE_TOP_N_SIGMA, COMMON_SAMPLER_TYPE_TOP_K, COMMON_SAMPLER_TYPE_TYPICAL_P, COMMON_SAMPLER_TYPE_TOP_P, 
  COMMON_SAMPLER_TYPE_MIN_P, COMMON_SAMPLER_TYPE_XTC, COMMON_SAMPLER_TYPE_TEMPERATURE}

```
```c++
struct common_params {
    ...
    struct common_params_sampling    sampling;
    ...
```
```c++
struct common_params_sampling {
    ...
    std::vector<enum common_sampler_type> samplers = {
        COMMON_SAMPLER_TYPE_PENALTIES,
        COMMON_SAMPLER_TYPE_DRY,
        COMMON_SAMPLER_TYPE_TOP_N_SIGMA,
        COMMON_SAMPLER_TYPE_TOP_K,
        COMMON_SAMPLER_TYPE_TYPICAL_P,
        COMMON_SAMPLER_TYPE_TOP_P,
        COMMON_SAMPLER_TYPE_MIN_P,
        COMMON_SAMPLER_TYPE_XTC,
        COMMON_SAMPLER_TYPE_TEMPERATURE,
    };
    ...
```
For this to work I think we need to order the samplers so that those that
expect raw logits come first and those that expect probabilities come last.

