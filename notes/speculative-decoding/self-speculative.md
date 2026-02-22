### Self-Speculative decoding
Self-speculative decoding is a technique where the insight is that we don't need
all of the model layers to get a useful draft prediction, but can get away with
only using a few. So instead of processing the full say 42 layers of the model
it might exit early after say 8 layers to get a draft prediction. It takes the
output of the hidden state at layer 8, and applies a classification head.

So here there is no need for a second model nor do we have to train any additional
layers/models.

### n-gram Speculative Decoding
n-gram in this context refers to a simplified "drafting" method that uses pattern
matching instead of a neural network to generate draft tokens. So this works by
looking backwards through the tokens that have already been generated (or the
initial prompt) and finding sequences of n tokens (n-grams) that have occurred
together previously. It then selects the prediction of those n-grams as the next
token(s) to draft.

In certain scenarious like coding where we often have repeated patterns this can
work quite well.

So we specify the number of of token to match, and also how many tokens after
a match to include in the draft prediction.

In llama.cpp there are difference n-gram based implementations which is chosen
based on the parameter specified to `common_speculative_init`:
```c++
    struct common_speculative * spec = common_speculative_init(params.speculative, ctx_tgt);
```
```c++
common_speculative * common_speculative_init(common_params_speculative & params, llama_context * ctx_tgt);
```
params can be found in common.h:
```c++
struct common_params {
    struct common_params_speculative speculative;
    ...
};
```

```c++
struct common_params_speculative {
    common_speculative_type type = COMMON_SPECULATIVE_TYPE_NONE; // type of speculative decoding
```

```c++
struct common_params_speculative {
    common_speculative_type type = COMMON_SPECULATIVE_TYPE_NONE; // type of speculative decoding
```
The following types are current available, though note that Eagle3 is not
implemented yet but there is an open draft pr for it:
```c++
enum common_speculative_type {
    COMMON_SPECULATIVE_TYPE_NONE,          // no speculative decoding
    COMMON_SPECULATIVE_TYPE_DRAFT,         // draft model
    COMMON_SPECULATIVE_TYPE_EAGLE3,        // eagle draft model
    COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE,  // simple self-speculative decoding
    COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K,   // self-speculative decoding with n-gram keys only
    COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V, // self-speculative decoding with n-gram keys and 4 m-gram values
    COMMON_SPECULATIVE_TYPE_NGRAM_MOD,
    COMMON_SPECULATIVE_TYPE_NGRAM_CACHE,   // self-speculative decoding with 3-level n-gram cache
    COMMON_SPECULATIVE_TYPE_COUNT          // number of types, unknown type
};
```
SO DRAFT would be the standard speculative decoding with a separate draft model.

Lets take a closer look at the init function and what it does:
```c++
common_speculative * common_speculative_init(common_params_speculative & params, llama_context * ctx_tgt) {
    llama_context * ctx_dft = nullptr;
    if (params.model_dft) {
        ctx_dft = llama_init_from_model(params.model_dft, params.cparams_dft);
        if (ctx_dft == nullptr) {
            LOG_ERR("%s", "failed to create draft context\n");
            return nullptr;
        }
    }
```
So a draft model is specified in the params it will be loaded and initalized 
just like any normal model.

Multiple speculative configs could be specified in the params and the following
code selects the best option:
```c++
    std::vector<common_speculative_config> configs = {}; // list of speculative configs to try
    {
        bool has_draft = !params.mparams_dft.path.empty();
        bool has_draft_eagle3 = false; // TODO PR-18039: if params.speculative.eagle3

        bool has_ngram_cache   = (params.type == COMMON_SPECULATIVE_TYPE_NGRAM_CACHE);
        bool has_ngram_simple  = (params.type == COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE);
        bool has_ngram_map_k   = (params.type == COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K);
        bool has_ngram_map_k4v = (params.type == COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V);
        bool has_ngram_mod     = (params.type == COMMON_SPECULATIVE_TYPE_NGRAM_MOD);
    

        if (has_ngram_simple) {
            // This implementation can guess a lot of tokens without any draft model.
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE, params));
        }
        if (has_ngram_map_k) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K, params));
        }
        if (has_ngram_map_k4v) {
            // This implementation can guess tokens with high acceptance rate but is more expensive.
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V, params));
        }
    }
```
The above are just adding new common_speculative_config to the configs vector:
```c++
struct common_speculative_config {
    common_speculative_type type;
    common_params_speculative params;
```

There is some additional handling for ngram_mod (modulus, as is the hashing method
used ?):
```c++
        if (has_ngram_mod) {
            // shared instance for all speculative decoding contexts
            if (!params.ngram_mod) {
                params.ngram_mod = std::make_shared<common_ngram_mod>(params.ngram_size_n, 4*1024*1024);

                LOG_INF("%s: initialized ngram_mod with n=%d, size=%zu (%.3f MB)\n", __func__,
                        params.ngram_size_n, params.ngram_mod->size(),
                        (float)(params.ngram_mod->size_bytes())/1024/1024);

                if (params.ngram_size_n < 16) {
                    LOG_WRN("%s: ngram_mod n=%d is too small - poor quality is possible, see: https://github.com/ggml-org/llama.cpp/pull/19164\n", __func__, params.ngram_size_n);
                }
            }

            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_MOD, params));
        }
```
So this takes the number of token to match (N). But what is the size? This is the
size of the entries vector in the common_ngram_mod struct:
```c++
// basic n-gram hasher
struct common_ngram_mod {
    using entry_t = int32_t;

    static constexpr entry_t EMPTY = -1;

    common_ngram_mod(uint16_t n, size_t size);

    size_t  idx(const entry_t * tokens) const;
    void    add(const entry_t * tokens);
    entry_t get(const entry_t * tokens) const; // return -1 if not found

    void reset();

    size_t get_n()    const;
    size_t get_used() const;

    size_t size()       const;
    size_t size_bytes() const;

private:
    size_t n; // ngram size to hash

    size_t used;

    std::vector<entry_t> entries;
};
```
```c++
common_ngram_mod::common_ngram_mod(uint16_t n, size_t size) : n(n), used(0) {
    entries.resize(size);

    reset();
}
```
TODO: link or return to the implementation of this later.
After that in init we have:
```c++
        if (has_ngram_cache) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_CACHE, params));
        }
        if (has_draft) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_DRAFT, params));
        }
        if (has_draft_eagle3) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_EAGLE3, params));
        }
```
And these are also just setting the type and params in the common_speculative_config
struct.
Next, the configurations will be iterated over:
```c++
    std::vector<std::unique_ptr<common_speculative_state>> impls = {};

    for (const common_speculative_config & config : configs) {

        switch (config.type) {
            case COMMON_SPECULATIVE_TYPE_NONE:
                break;
            case COMMON_SPECULATIVE_TYPE_DRAFT: {
                impls.push_back(std::make_unique<common_speculative_state_draft>(config.type,
                    /* .ctx_tgt      = */ ctx_tgt,
                    /* .ctx_dft      = */ ctx_dft,
                    /* .replacements = */ params.replacements
                ));
                break;
            }
            ...
```
So here we can see the usage of the params and for which speculative type they
are used. There is a case for each type and there are corresponding
common_speculative_state implementations for each type as well.
The init function then returns a common_speculative struct:
```c++
    auto * result = new common_speculative {
        /* .impls = */ std::move(impls)
    };

    return result;
```
So that was when we called common_speculative_init, and this would then be
followed by common_speculative_begin:
```c++
    struct common_speculative * spec = common_speculative_init(params.speculative, ctx_tgt);
    common_speculative_begin(spec, prompt_tgt);
```
This is called before any decoding begins so that the speculative method can
learn from the initial prompt.
```c++
void common_speculative_begin(common_speculative * spec, const llama_tokens & prompt) {
    if (spec == nullptr) {
        return;
    }

    for (auto & impl : spec->impls) {
        common_time_meas tm(impl->t_begin_us, !impl->gen_perf);
        impl->begin(prompt);
        impl->n_call_begin++;
    }
}
```
```c++
struct common_speculative_state_ngram_mod : public common_speculative_state {
    common_ngram_mod & mod;
    ...

    void begin(const llama_tokens & prompt) override {
        i_last = 0;

        n_draft_last = 0;

        const size_t n = mod.get_n();

        if (prompt.size() < n) {
            return;
        }

        for (size_t i = 0; i < prompt.size() - n; ++i) {
            mod.add(prompt.data() + i);
        }

        i_last = prompt.size() - n;

        const double f = (double)mod.get_used() / (double)mod.size();
        LOG_INF("%s: ngram_mod occupancy = %zu/%zu (%.2f)\n", __func__, mod.get_used(), mod.size(), f);

        constexpr double f_thold = 0.25;
        if (f > f_thold) {
            LOG_WRN("%s: ngram_mod occupancy %.2f exceeds threshold (%.2f) - resetting\n", __func__, f, f_thold);

            mod.reset();
        }
    }
```
So at the moment the ngram_mod is hooked up to llama-server and we can step
through it in a debugger is we specify `--spec-type ngram-mod
```console
(lldb) p n
(const size_t) 12
(lldb) p prompt.size()
(std::vector<int>::size_type) 36
```
So this function is iterating over all 36-12=24 (and only < 24) so this will
iterate over 23 token ids. So we are calling this with a pointer to the first
token id in the prompt which will hash the first 12 tokens. For the next iteration
it will do the same for the second token id and the following 12 tokens. This is
like a sliding window of size 12 moving across the prompt.
```c++
        for (size_t i = 0; i < prompt.size() - n; ++i) {
            mod.add(prompt.data() + i);
        }
```
```c++
void common_ngram_mod::add(const entry_t * tokens) {
    const size_t i = idx(tokens);

    if (entries[i] == EMPTY) {
        used++;
    }

    entries[i] = tokens[n];
}
```
So first idx is called which is what hashes the passed in tokens
```c++
size_t common_ngram_mod::idx(const entry_t * tokens) const {
    size_t res = 0;

    for (size_t i = 0; i < n; ++i) {
        res = res*6364136223846793005ULL + tokens[i];
    }

    res = res % entries.size();

    return res;
}
```

_wip_


### n-gram simple
TODO: 

### n-gram map k
TODO:

### n-gram map k4v
TODO:

### n-gram mod
TODO:

### n-gram cache
TODO:
