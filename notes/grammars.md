## Grammars

Example of using a grammar file:
```console
./build/bin/llama-cli -m models/llama-2-7b.Q4_K_M.gguf --no-warmup \
    --prompt '"What is capital of Sweden?"' -n 40  -ngl 99 \
    --grammar-file grammars/json.gbnf

"What is capital of Sweden?"{ "Sweden" : "Stockholm" }
[end of text]
```

After a batch has been decoded by llama_decode it will most often have produces
logits over/for the vocabulary of the model. 
```c++
            const llama_token id = common_sampler_sample(smpl, ctx, -1);
```
This function is defined in `common/sampling.h`:
```c++
// if grammar_first is true, the grammar is applied before the samplers (slower)
// useful in cases where all the resulting candidates (not just the sampled one) must fit the grammar
//
llama_token common_sampler_sample(struct common_sampler * gsmpl, struct llama_context * ctx, int idx, bool grammar_first = false);
```

```c++
llama_token common_sampler_sample(struct common_sampler * gsmpl, struct llama_context * ctx, int idx, bool grammar_first) {
    gsmpl->set_logits(ctx, idx);

```
```console
(gdb) ptype gsmpl
type = struct common_sampler {
    common_params_sampling params;
    llama_sampler *grmr;
    llama_sampler *chain;
    ring_buffer<int> prev;
    std::vector<llama_token_data> cur;
    llama_token_data_array cur_p;

    void set_logits(llama_context *, int);
} *
```
```c++
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
This is setting the current probabilities to the logits of the last prediction
performed.
```console
(gdb) ptype cur_p
type = struct llama_token_data_array {
    llama_token_data *data;
    size_t size;
    int64_t selected;
    bool sorted;
}
(gdb) ptype llama_token_data
type = struct llama_token_data {
    llama_token id;
    float logit;
    float p;
}
(gdb) p cur_p
$4 = {data = 0x55555e08fdf0, size = 32000, selected = -1, sorted = false}
```
After the logits have been set we have the following code:
```c++
    auto & grmr  = gsmpl->grmr;
    auto & chain = gsmpl->chain;
    auto & cur_p = gsmpl->cur_p; // initialized by set_logits

    if (grammar_first) {
        llama_sampler_apply(grmr, &cur_p);
    }

    llama_sampler_apply(chain, &cur_p);
```

llama_sampler_apply can be found in `src/llama-sampling.cpp`:
```c++
void llama_sampler_apply(struct llama_sampler * smpl, struct llama_token_data_array * cur_p) {
    GGML_ASSERT(smpl->iface->apply);
    smpl->iface->apply(smpl, cur_p);
}
```
```c++
static void llama_sampler_chain_apply(struct llama_sampler * smpl, llama_token_data_array * cur_p) {
    auto * chain = (llama_sampler_chain *) smpl->ctx;

    time_meas tm(chain->t_sample_us, chain->params.no_perf);

    for (auto * smpl : chain->samplers) {
        llama_sampler_apply(smpl, cur_p);
    }
}
```
So this is going to call all the samplers in the chain. In this case there
are 10 samplers in the chain.

The first one in the chain is a logit bias sampler.
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
````
What this sampler does enables a way to modify how likely certain tokens are to
be selected during sampling. We can see above that is stored in `ctx` which
I found somewhat confusing (the naming as I though this was a ggml_context or a
llama_context) but if we look closer it is actual of type
`llama_sampler_logit_bias`. 
```console
(gdb) ptype llama_sampler_logit_bias
type = struct llama_sampler_logit_bias {
    const int32_t n_vocab;
    const std::vector<llama_logit_bias> logit_bias;
    std::vector<llama_logit_bias> to_search;
}
```
So bascially what this is doing is that it is iterating over all the logit biases
and there is a match in the curren probabilities data for this token id the
bias will be applied. So where is these logit biases coming from?   For this we
need to look back in main.cpp:
```c++
    smpl = common_sampler_init(model, sparams);
```
```c++
struct common_sampler * common_sampler_init(const struct llama_model * model, const struct common_params_sampling & params) {
    const llama_vocab * vocab = llama_model_get_vocab(model);

    llama_sampler_chain_params lparams = llama_sampler_chain_default_params();

    lparams.no_perf = params.no_perf;

    std::vector<const char *> trigger_words;
    trigger_words.reserve(params.grammar_trigger_words.size());
    for (const auto & str : params.grammar_trigger_words) {
        trigger_words.push_back(str.word.c_str());
    }
```
What is the trigger words used for?  They are used below in the lazy grammar
initialization. More on this later.

Following that we have:
```c++

    struct llama_sampler * grmr;
    if (params.grammar.compare(0, 11, "%llguidance") == 0) {
#ifdef LLAMA_USE_LLGUIDANCE
        grmr = llama_sampler_init_llg(vocab, "lark", params.grammar.c_str());
#else
        GGML_ABORT("llguidance (cmake -DLLAMA_LLGUIDANCE=ON) is not enabled");
#endif // LLAMA_USE_LLGUIDANCE
    } else {
        grmr = params.grammar_lazy
             ? llama_sampler_init_grammar_lazy(vocab, params.grammar.c_str(), "root",
                                               trigger_words.data(), trigger_words.size(),
                                               params.grammar_trigger_tokens.data(), params.grammar_trigger_tokens.size())
             :      llama_sampler_init_grammar(vocab, params.grammar.c_str(), "root");
    }
```
Notice that this is checking the first 11 characters of the grammar string to
see if it starts with `%llguidance`. This is not the case in this session:
```console
(gdb) p params.grammar
$12 = "root   ::= object\nvalue  ::= object | array | string | number | (\"true\" | \"false\" | \"null\") ws\n\nobject ::=\n  \"{\" ws (\n", ' ' <repeats 12 times>, "string \":\" ws value\n    (\",\" ws string \":\" ws value)*\n  )? \"}\" ws\n\narr"..
```

```c++

    auto * result = new common_sampler {
        /* .params = */ params,
        /* .grmr   = */ grmr,
        /* .chain  = */ llama_sampler_chain_init(lparams),
        /* .prev   = */ ring_buffer<llama_token>(std::max(32, params.n_prev)),
        /* .cur    = */ {},
        /* .cur_p  = */ {},
    };
    ...
```
So lets take a closer look at the sampler initializtion which is our case
is not using the lazy samling:
```c++
    } else {
        grmr = params.grammar_lazy
             ? llama_sampler_init_grammar_lazy(vocab, params.grammar.c_str(), "root",
                                               trigger_words.data(), trigger_words.size(),
                                               params.grammar_trigger_tokens.data(), params.grammar_trigger_tokens.size())
             :      llama_sampler_init_grammar(vocab, params.grammar.c_str(), "root");
    }
```
So this will call `llama_sampler_init_grammar`.
```c++
struct llama_sampler * llama_sampler_init_grammar(
        const struct llama_vocab * vocab,
                      const char * grammar_str,
                      const char * grammar_root) {
    return llama_sampler_init_grammar_impl(vocab, grammar_str, grammar_root, /* lazy= */ false, nullptr, 0, nullptr, 0);
}
```
So this is delegating and passing through the vocab, the grammar as a string
and the root of the grammar (there might be more than one I think so this is a
way to specify which one to use). The reset are flags for lazy initialization
and set to nullptr and 0.

```c++
static struct llama_sampler * llama_sampler_init_grammar_impl(
        const struct llama_vocab * vocab,
                      const char * grammar_str,
                      const char * grammar_root,
                              bool lazy,
                     const char ** trigger_words,
                            size_t num_trigger_words,
               const llama_token * trigger_tokens,
                            size_t num_trigger_tokens) {
    auto * ctx = new llama_sampler_grammar;

    if (grammar_str != nullptr && grammar_str[0] != '\0') {
        *ctx = {
            /* .vocab        = */ vocab,
            /* .grammar_str  = */ grammar_str,
            /* .grammar_root = */ grammar_root,
            /* .grammar      = */ llama_grammar_init_impl(vocab, grammar_str, grammar_root, lazy, trigger_words, num_trigger_words, trigger_tokens, num_trigger_tokens),
        };
    } else {
        *ctx = {
            /* .vocab        = */ vocab,
            /* .grammar_str  = */ {},
            /* .grammar_root = */ {},
            /* .grammar      = */ nullptr,
        };
    }

    return llama_sampler_init(
        /* .iface = */ &llama_sampler_grammar_i,
        /* .ctx   = */ ctx
    );
}

static struct llama_sampler_i llama_sampler_grammar_i = {
    /* .name   = */ llama_sampler_grammar_name,
    /* .accept = */ llama_sampler_grammar_accept_impl,
    /* .apply  = */ llama_sampler_grammar_apply,
    /* .reset  = */ llama_sampler_grammar_reset,
    /* .clone  = */ llama_sampler_grammar_clone,
    /* .free   = */ llama_sampler_grammar_free,
};
```
`llama_sampler_grammar` looks like this:
```console
(gdb) ptype llama_sampler_grammar
type = struct llama_sampler_grammar {
    const llama_vocab *vocab;
    std::string grammar_str;
    std::string grammar_root;
    llama_grammar *grammar;
}
```
And notice that we are initializing the grammer above by calling
`llama_grammar_init_impl`.
After this function returns we will be back in `common/sampling.cpp`:
```c++
    auto * result = new common_sampler {
        /* .params = */ params,
        /* .grmr   = */ grmr,
        /* .chain  = */ llama_sampler_chain_init(lparams),
        /* .prev   = */ ring_buffer<llama_token>(std::max(32, params.n_prev)),
        /* .cur    = */ {},
        /* .cur_p  = */ {},
    };

    llama_sampler_chain_add(result->chain,
            llama_sampler_init_logit_bias(
                llama_vocab_n_tokens(vocab),
                params.logit_bias.size(),
                params.logit_bias.data()));
```
And now we can finally see where the logit biases are coming from. The are
configured in the `common_params_sampling` and is a parameter to llama-cli:
```console
$ ./build/bin/llama-cli --help | grep logit
-l,    --logit-bias TOKEN_ID(+/-)BIAS   modifies the likelihood of token appearing in the completion,
                                        i.e. `--logit-bias 15043+1` to increase likelihood of token ' Hello',
                                        or `--logit-bias 15043-1` to decrease likelihood of token ' Hello'
```

### Low Level Grammar Initialization
[llguidance.md](./llguidance.md)

### Lazy Grammar Initialization
TODO:
