## Grammars

Example of using a grammar file:
```console
./build/bin/llama-cli -m models/llama-2-7b.Q4_K_M.gguf --no-warmup \
    --prompt '"What is capital of Sweden?"' -n 40  -ngl 99 \
    --grammar-file grammars/json.gbnf

"What is capital of Sweden?"{ "Sweden" : "Stockholm" }
[end of text]
```

After a batch has been decoded by llama_decode it will most often have produced
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
This is setting the current probabilities (cur_p) to the logits of the last
prediction performed.
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
```
What this sampler does is it enables a way to modify how likely certain tokens
are to be selected during sampling. We can see above that the biases are stored
in `ctx` which I found somewhat confusing (the naming as I thougth this was a
ggml_context or a llama_context) but if we look closer it is actually of type
`llama_sampler_logit_bias`: 
```console
(gdb) ptype llama_sampler_logit_bias
type = struct llama_sampler_logit_bias {
    const int32_t n_vocab;
    const std::vector<llama_logit_bias> logit_bias;
    std::vector<llama_logit_bias> to_search;
}
```
So bascially what this is doing is that it is iterating over all the logit biases
and if there is a match in the current probabilities data for this token id, the
bias will be applied (added). So where is these logit biases coming from?   For
this we need to look back in main.cpp:
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
And the other samplers are added in a similar fashion, for example:
```c++
    if (params.mirostat == 0) {
        for (const auto & cnstr : params.samplers) {
            switch (cnstr) {
                case COMMON_SAMPLER_TYPE_DRY:
                    {
                        std::vector<const char *> c_breakers;
                        c_breakers.reserve(params.dry_sequence_breakers.size());
                        for (const auto & str : params.dry_sequence_breakers) {
                            c_breakers.push_back(str.c_str());
                        }

                        llama_sampler_chain_add(result->chain, llama_sampler_init_dry      (vocab, llama_model_n_ctx_train(model), params.dry_multiplier, params.dry_base, params.dry_allowed_length, params.dry_penalty_last_n, c_breakers.data(), c_breakers.size()));
                    }
                    break;
                case COMMON_SAMPLER_TYPE_TOP_K:
                ...
```
TODO: Look into the DRY sampler.

After `llama_decode` has been called `common_sample_sample` will be called which
we discussed above:
```c++
            const llama_token id = common_sampler_sample(smpl, ctx, -1);
```
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

static void llama_sampler_chain_apply(struct llama_sampler * smpl, llama_token_data_array * cur_p) {
    auto * chain = (llama_sampler_chain *) smpl->ctx;

    time_meas tm(chain->t_sample_us, chain->params.no_perf);

    for (auto * smpl : chain->samplers) {
        llama_sampler_apply(smpl, cur_p);
    }
}
```
And like we saw before we will iterate over all the samplers in the chain.
So this will first call the logit bias sampler. One thing to notice is that
`cur_p` are the current logits and the logit biases are added to these logits.
The next sampler will be using these now possibly modified logits. And the
next sampler will be the repition penalty sampler, and it may modify the logits
further and so on.

After this one of the tokens will have been selected by the samplers::
```c++
llama_token common_sampler_sample(struct common_sampler * gsmpl, struct llama_context * ctx, int idx, bool grammar_first) {
    ...

    const llama_token id = cur_p.data[cur_p.selected].id;

    // check if it the sampled token fits the grammar
    {
        llama_token_data       single_token_data       = { id, 1.0f, 0.0f };
        llama_token_data_array single_token_data_array = { &single_token_data, 1, -1, false };

        llama_sampler_apply(grmr, &single_token_data_array);

        const bool is_valid = single_token_data_array.data[0].logit != -INFINITY;
        if (is_valid) {
            return id;
        }
    }
```
Note here that `llama_sampler_apply` is now being called with the grammar sampler
and that it is passed a single token data array. So the grammar samplers will
be given the token selected by the samplers.
```c++
static void llama_sampler_grammar_apply(struct llama_sampler * smpl, llama_token_data_array * cur_p) {
    auto * ctx = (llama_sampler_grammar *) smpl->ctx;
    if (ctx->grammar) {
        llama_grammar_apply_impl(*ctx->grammar, cur_p);
    }
}
```
The grammar will check that the token is valid and if not the grammar sampler
will set `single_token_data_array.data[0].logit` to `-INFINITY`. 

So the first time that llama_sampler_apply(grmr...) is called it is done so with
a single token, the one that the samplers selected. If this was not a valid
token according to the grammer, then it will try again but this time it will
have access to all the logits. Once it has a valid token the other samplers will
also be run.

To be clear about this the grammar sampler will set the logit to -INFINITY if
it is invalid:
```c++
    const auto rejects = llama_grammar_reject_candidates(grammar.rules, grammar.stacks, candidates_grammar);
    for (const auto & reject : rejects) {
-->    cur_p->data[reject.index].logit = -INFINITY;
    }
```
So it modifies the logits which the other samplers then also use. So is kind of
like filtering out the tokens that don't match the grammar.

Now back in main.cpp we have just called `common_sampler_sample` and will
proceed to call `common_sampler_accept`:
```c++
            const llama_token id = common_sampler_sample(smpl, ctx, -1);

            common_sampler_accept(smpl, id, /* accept_grammar= */ true);
```

```c++
void common_sampler_accept(struct common_sampler * gsmpl, llama_token token, bool accept_grammar) {
    if (accept_grammar) {
        llama_sampler_accept(gsmpl->grmr, token);
    }

    llama_sampler_accept(gsmpl->chain, token);

    gsmpl->prev.push_back(token);
}
```
So lets take a look at what `llama_sampler_accept` for the grammar sampler:
```c++
void llama_sampler_accept(struct llama_sampler * smpl, llama_token token) {
    if (smpl->iface->accept) {
        smpl->iface->accept(smpl, token);
    }
}
```
This will end up in:
```c++
static void llama_sampler_grammar_accept_impl(struct llama_sampler * smpl, llama_token token) {
    auto * ctx = (llama_sampler_grammar *) smpl->ctx;
    if (ctx->grammar) {
        llama_grammar_accept_impl(*ctx->grammar, token);
    }
}
```
Lets just inspect some variables to orient ourselves:
```console
(gdb) p token
$18 = 29912

(gdb) p grammar.vocab.pimpl->id_to_token[token]
$20 = {text = "{", score = -29653, attr = LLAMA_TOKEN_ATTR_NORMAL}

(gdb) p grammar.vocab->token_to_piece(token)
$21 = "{"
```
And recall that we are using a JSON grammer so the `{` token is the start of
an object. 

So lets looks what accept does:
```c++
void llama_grammar_accept_impl(struct llama_grammar & grammar, llama_token token) {
    GGML_ASSERT(grammar.vocab != nullptr);

    const auto & piece = grammar.vocab->token_to_piece(token);

    if (grammar.awaiting_trigger) {
        if (std::find(grammar.trigger_tokens.begin(), grammar.trigger_tokens.end(), token) != grammar.trigger_tokens.end()) {
            grammar.awaiting_trigger = false;
            grammar.trigger_buffer.clear();
            llama_grammar_accept_str(grammar, piece);
            LLAMA_LOG_DEBUG("Grammar triggered on token %u (`%s`)", token, piece.c_str());
            return;
        } else {
            // TODO: consider a smarter incremental substring search algorithm (store last position to search from).
            grammar.trigger_buffer += piece;
            for (const auto & word : grammar.trigger_words) {
                auto pos = grammar.trigger_buffer.find(word);
                if (pos != std::string::npos) {
                    grammar.awaiting_trigger = false;
                    auto constrained_str = grammar.trigger_buffer.substr(pos);
                    grammar.trigger_buffer.clear();
                    llama_grammar_accept_str(grammar, constrained_str);
                    LLAMA_LOG_DEBUG("Grammar triggered on word `%s`", word.c_str());
                    return;
                }
            }
            LLAMA_LOG_DEBUG("Grammar still awaiting trigger after token %d (`%s`) (buffer: `%s`)\n", token, piece.c_str(), grammar.trigger_buffer.c_str());
            return;
        }
    }

    if (grammar.vocab->is_eog(token)) {
        for (const auto & stack : grammar.stacks) {
            if (stack.empty()) {
                return;
            }
        }
        GGML_ABORT("fatal error");
    }

    llama_grammar_accept_str(grammar, piece);
}
```
TODO: Look into awaiting trigger.
In our case `awaiting_trigger` is false so we will go to the next if statement.
And this token is not an end of generation token that if block will be skipped
and we will call `llama_grammer_accept_str`:
```c++
void llama_grammar_accept_str(struct llama_grammar & grammar, const std::string & piece) {
    // Note terminating 0 in decoded string
    const auto   decoded     = decode_utf8(piece, grammar.partial_utf8);
    const auto & code_points = decoded.first;

    for (auto it = code_points.begin(), end = code_points.end() - 1; it != end; ++it) {
        llama_grammar_accept(&grammar, *it);
    }

    grammar.partial_utf8 = decoded.second;
    if (grammar.stacks.empty()) {
        throw std::runtime_error("Unexpected empty grammar stack after accepting piece: " + piece);
    }
}
```
```console
(gdb) p decoded
$25 = {first = std::vector of length 2, capacity 2 = {123, 0}, second = {value = 123, n_remain = 0}}

(gdb) p code_points.size()
$31 = 2
```
So this means that this piece is a single unicode character, 123 followed by
a terminating null value. The second part of the pair is the partial utf8
structure which is just the same unicode character and the number of remaining
bytes need (none in this case).

Now, this then the code will iterate over the 2 code points and call:
```c++
void llama_grammar_accept(struct llama_grammar * grammar, uint32_t chr) {
    llama_grammar_stacks stacks_new;
    stacks_new.reserve(grammar->stacks.size());

    for (const auto & stack : grammar->stacks) {
        if (stack.empty()) {
            continue;
        }

        auto match = llama_grammar_match_char(stack.back(), chr);
        if (match.first) {
            const llama_grammar_element * pos = match.second;

            // update top of stack to next element, if any
            llama_grammar_stack new_stack(stack.begin(), stack.end() - 1);
            if (!llama_grammar_is_end_of_sequence(pos)) {
                new_stack.push_back(pos);
            }
            llama_grammar_advance_stack(grammar->rules, new_stack, stacks_new);
        }
    }

    grammar->stacks = std::move(stacks_new);
}
```
So that was the grammar, but we also have the samplers chain:
```c++
void common_sampler_accept(struct common_sampler * gsmpl, llama_token token, bool accept_grammar) {
    if (accept_grammar) {
        llama_sampler_accept(gsmpl->grmr, token);
    }

    llama_sampler_accept(gsmpl->chain, token);

    gsmpl->prev.push_back(token);
}
```
Some samplers need to keep track of previous tokens, for example the repetition
penalty sample would need to know which tokens it has seen before:
```c++
static void llama_sampler_penalties_accept(struct llama_sampler * smpl, llama_token token) {
    auto * ctx = (llama_sampler_penalties *) smpl->ctx;
    if (ctx->penalty_last_n == 0) {
        return;
    }

    ctx->token_count[token]++;

    // if the ring buffer is full, remove the oldest token
    if (ctx->prev.size() >= (size_t) ctx->penalty_last_n) {
        const auto old = ctx->prev.front();

        ctx->token_count[old]--;
        if (ctx->token_count[old] == 0) {
            ctx->token_count.erase(old);
        }
    }

    ctx->prev.push_back(token);
}
```
So this will insert or updated the token count for the token that was just
selected:
```console
(gdb) p ctx->token_count
$47 = std::unordered_map with 0 elements
```
So this is what `accept` means. It is the samplers that get a chance to update
their internal state based on the token that was just selected. And not all
sampler may need this but some do.


### Low Level Grammar Initialization
[llguidance.md](./llguidance.md)

### Lazy Grammar Initialization
Going through the server code I came accross this parameter/option. But before
we look at the lazy part lets just go over an example of using a grammar.
```console
curl -fsS \
  --url http://127.0.0.1:8080/completion \
  --header "Content-Type: application/json" \
  --data '{"prompt": "What is LoRA?", "grammar": "<grammar goes here"}' | jq
```
So we would specify the grammar in the JSON payload. Now this would affect every
token generated which would have to adhere to the the grammar in question. And
by default the grammar is applied after the samplers have selected a token.

What `grammar_lazy` does is that it will only apply the grammar if we encounter
specific tokens generated by the samplers. This fields should be a list of
object that match the following struct in `common/common.h`:
```c++
struct common_grammar_trigger {
    std::string word;
    bool at_start;
};
```
For example:
```json
{
  "grammar_triggers": [
    {"word": "{", "at_start": false},
    {"word": "[", "at_start": false}
  ]
}
```
So if/when the model generates a `{` or a `[` token the grammar will be applied
but not otherwise.

So with `grammar_lazy` set to true the sampler will be initialized by:
```c++
bool launch_slot_with_task(server_slot & slot, const server_task & task) {
    ...
            slot.smpl = common_sampler_init(model, slot.params.sampling);
            ...
}
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
And in this case `llama_sampler_init_grammar_lazy` will be called:
```c++
struct llama_sampler * llama_sampler_init_grammar_lazy(
        const struct llama_vocab * vocab,
                      const char * grammar_str,
                      const char * grammar_root,
                     const char ** trigger_words,
                            size_t num_trigger_words,
               const llama_token * trigger_tokens,
                            size_t num_trigger_tokens) {
    return llama_sampler_init_grammar_impl(vocab, grammar_str, grammar_root, /* lazy= */ true, trigger_words, num_trigger_words, trigger_tokens, num_trigger_tokens);
}

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
```
And that will end up in `llama_grammar_init_impl`:
```c++
struct llama_grammar * llama_grammar_init_impl(
        const struct llama_vocab * vocab,
                      const char * grammar_str,
                      const char * grammar_root,
                              bool lazy,
                     const char ** trigger_words,
                            size_t num_trigger_words,
               const llama_token * trigger_tokens,
                            size_t num_trigger_tokens) {
    ...

    std::vector<llama_token> vec_trigger_tokens;
    std::vector<std::string> vec_trigger_words;
    for (size_t i = 0; i < num_trigger_tokens; i++) {
        GGML_ASSERT(trigger_tokens != nullptr);
        vec_trigger_tokens.push_back(trigger_tokens[i]);
    }
    for (size_t i = 0; i < num_trigger_words; i++) {
        GGML_ASSERT(trigger_words != nullptr);
        vec_trigger_words.push_back(trigger_words[i]);
    }

    return new llama_grammar {
        vocab,
        std::move(vec_rules),
        std::move(stacks),
        /* .partial_utf8 = */     {},
        /* .lazy = */             lazy,
        /* .awaiting_trigger = */ lazy,
        /* .trigger_buffer = */   "",
        std::move(vec_trigger_tokens),
        std::move(vec_trigger_words),
    };
```
Notice that `awaiting_trigger` is set to `lazy/true`.

So if we set a breakpoint 
```console
(gdb) br server.cpp:361
(gdb) r
```
And then run the following bash script:
```console
GRAMMAR_CONTENT=$(cat grammars/json.gbnf)

JSON_PAYLOAD=$(jq -n \
  --arg prompt "Describe a cat in JSON format." \
  --arg grammar "$GRAMMAR_CONTENT" \
  '{
    prompt: $prompt,
    n_predict: 160,
    verbose: true,
    temperature: 0,
    grammar: $grammar,
    grammar_lazy: true,
    grammar_triggers: [
      { word: "{", at_start: false}
    ]
  }')

echo "JSON_PAYLOAD: $JSON_PAYLOAD"

curl -fsS \
  --url http://127.0.0.1:8080/completion \
  --header "Content-Type: application/json" \
  --data "$JSON_PAYLOAD" | jq
GRAMMAR_CONTENT=$(cat grammars/json.gbnf)
```
We can then step through the server code and inspect the values:
```console
(gdb) p trigger
$5 = {word = "{", at_start = false}
```

```c++
            const auto grammar_triggers = data.find("grammar_triggers");
            if (grammar_triggers != data.end()) {
                for (const auto & t : *grammar_triggers) {
                    common_grammar_trigger trigger;
                    trigger.word = t.at("word");
                    trigger.at_start = t.at("at_start");

                    auto ids = common_tokenize(vocab, trigger.word, /* add_special= */ false, /* parse_special= */ true);
                    if (ids.size() == 1) {
                        SRV_DBG("Grammar trigger token: %d (`%s`)\n", ids[0], trigger.word.c_str());
                        params.sampling.grammar_trigger_tokens.push_back(ids[0]);
                        params.sampling.preserved_tokens.insert(ids[0]);
                        continue;
                    }
                    SRV_DBG("Grammar trigger word: `%s`\n", trigger.word.c_str());
                    params.sampling.grammar_trigger_words.push_back(trigger);
            }
```
Notice that the above will tokenize the trigger word:
```console
(gdb) p ids
$7 = std::vector of length 1, capacity 1 = {426}
(gdb) p vocab->pimpl->id_to_token[ids[0]]
$9 = {text = "▁{", score = -167, attr = LLAMA_TOKEN_ATTR_NORMAL}
```
This token will be added to the `grammar_trigger_tokens` vector and also stored
in `preserved_tokens` which is a set of tokens that should not be modified by
the samplers (double check this). So this is all done when the server is parsing
the payload from the client (this is done in handle_completions_impl).
Also notice that if there is only a single token then the `grammer_trigger_words`
is not updated. It will only be updated if there are multiple tokens.

Later in `update_slots`, after `llama_decode` we then have:
```c++
    llama_token id = common_sampler_sample(slot.smpl, ctx, tok_idx);
```
And like we discussed above what will happen is that if `grammar_first` is false
(the default) the samplers will first be processed and have the chance to adjust
the logits. Following that, the token that the samplers selected will be passed
to the grammar sampler which is the part we are interested in here:
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

    GGML_ASSERT(cur_p.selected != -1 && "no selected token during sampling - check your sampling configuration");

    const llama_token id = cur_p.data[cur_p.selected].id;

    if (grammar_first) {
        return id;
    }

    // check if it the sampled token fits the grammar
    {
        llama_token_data       single_token_data       = { id, 1.0f, 0.0f };
        llama_token_data_array single_token_data_array = { &single_token_data, 1, -1, false };

        llama_sampler_apply(grmr, &single_token_data_array);

        const bool is_valid = single_token_data_array.data[0].logit != -INFINITY;
        if (is_valid) {
            return id;
        }
    }

    // resampling:
    // if the token is not valid, sample again, but first apply the grammar sampler and then the sampling chain
    gsmpl->set_logits(ctx, idx);

    llama_sampler_apply(grmr,  &cur_p);
    llama_sampler_apply(chain, &cur_p);

    GGML_ASSERT(cur_p.selected != -1 && "no selected token during re-sampling - check your sampling configuration");

    return cur_p.data[cur_p.selected].id;
}
```
In this case the following token was selected:
```console
(gdb) p id
$11 = 29889
(gdb) p ctx.model.vocab->pimpl->id_to_token[id]
$15 = {text = ".", score = -29630, attr = LLAMA_TOKEN_ATTR_NORMAL}
```
```c++
        llama_sampler_apply(grmr, &single_token_data_array);
```
```c++
static void llama_sampler_grammar_apply(struct llama_sampler * smpl, llama_token_data_array * cur_p) {
    auto * ctx = (llama_sampler_grammar *) smpl->ctx;
    if (ctx->grammar) {
        llama_grammar_apply_impl(*ctx->grammar, cur_p);
    }
}
```
And in `src/llama-grammer.cpp` we find:
```c++
void llama_grammar_apply_impl(const struct llama_grammar & grammar, llama_token_data_array * cur_p) {
    GGML_ASSERT(grammar.vocab != nullptr);

    if (grammar.awaiting_trigger) {
        return;
    }
    ...
```
Now, in our case `awaiting_trigger` is true so this will return to
`common_sampler_sample`:
```c++
        const bool is_valid = single_token_data_array.data[0].logit != -INFINITY;
        if (is_valid) {
            return id;
        }
```
And this will return us to `update_slots`:
```c++
                llama_token id = common_sampler_sample(slot.smpl, ctx, tok_idx);

                slot.i_batch = -1;

                common_sampler_accept(slot.smpl, id, true);
```
Now, this call will end up in `llama_grammar_accept_impl`, and recall that
`grammar.awaiting_trigger` is true:
```c++
void llama_grammar_accept_impl(struct llama_grammar & grammar, llama_token token) {
    GGML_ASSERT(grammar.vocab != nullptr);

    const auto & piece = grammar.vocab->token_to_piece(token);

    if (grammar.awaiting_trigger) {
        if (std::find(grammar.trigger_tokens.begin(), grammar.trigger_tokens.end(), token) != grammar.trigger_tokens.end()) {
            grammar.awaiting_trigger = false;
            grammar.trigger_buffer.clear();
            llama_grammar_accept_str(grammar, piece);
            LLAMA_LOG_DEBUG("Grammar triggered on token %u (`%s`)", token, piece.c_str());
            return;
        } else {
            // TODO: consider a smarter incremental substring search algorithm (store last position to search from).
            grammar.trigger_buffer += piece;
            for (const auto & word : grammar.trigger_words) {
                auto pos = grammar.trigger_buffer.find(word);
                if (pos != std::string::npos) {
                    grammar.awaiting_trigger = false;
                    auto constrained_str = grammar.trigger_buffer.substr(pos);
                    grammar.trigger_buffer.clear();
                    llama_grammar_accept_str(grammar, constrained_str);
                    LLAMA_LOG_DEBUG("Grammar triggered on word `%s`", word.c_str());
                    return;
                }
            }
            LLAMA_LOG_DEBUG("Grammar still awaiting trigger after token %d (`%s`) (buffer: `%s`)\n", token, piece.c_str(), grammar.trigger_buffer.c_str());
            return;
        }
    }

    if (grammar.vocab->is_eog(token)) {
        for (const auto & stack : grammar.stacks) {
            if (stack.empty()) {
                return;
            }
        }
        GGML_ABORT("fatal error");
    }

    llama_grammar_accept_str(grammar, piece);
}
```



Back in update_slots we will then have the following call:
```c++
                completion_token_output result;
                result.tok          = id;
                result.text_to_send = common_token_to_piece(ctx, result.tok, accept_special_token(slot, result.tok));
                result.prob         = 1.0f; // TODO: set it here instead of doing inside populate_token_probs

                if (slot.params.sampling.n_probs > 0) {
                    populate_token_probs(slot, result, slot.params.post_sampling_probs, params_base.special, tok_idx);
                }

                if (!process_token(result, slot)) {
```
Lets take a look at `process_token`:
```c++
    bool process_token(completion_token_output & result, server_slot & slot) {
        // remember which tokens were sampled - used for repetition penalties during sampling
        const std::string token_str = result.text_to_send;
        slot.sampled = result.tok;

        slot.generated_text += token_str;
        if (slot.params.return_tokens) {
            slot.generated_tokens.push_back(result.tok);
        }
        slot.has_next_token = true;
```

If we inspect the following call when the trigger token matches, we ill call
`llama_gramar_accept_str` with the piece (the string representation of the token):
```c++
void llama_grammar_accept_impl(struct llama_grammar & grammar, llama_token token) {
    GGML_ASSERT(grammar.vocab != nullptr);

    const auto & piece = grammar.vocab->token_to_piece(token);

    if (grammar.awaiting_trigger) {
        if (std::find(grammar.trigger_tokens.begin(), grammar.trigger_tokens.end(), token) != grammar.trigger_tokens.end()) {
            grammar.awaiting_trigger = false;
            grammar.trigger_buffer.clear();
            llama_grammar_accept_str(grammar, piece);
            LLAMA_LOG_DEBUG("Grammar triggered on token %u (`%s`)", token, piece.c_str());
            return;
```
So the above will call:
```c++
void llama_grammar_accept_str(struct llama_grammar & grammar, const std::string & piece) {
    // Note terminating 0 in decoded string
    const auto   decoded     = decode_utf8(piece, grammar.partial_utf8);
    const auto & code_points = decoded.first;

    for (auto it = code_points.begin(), end = code_points.end() - 1; it != end; ++it) {
        llama_grammar_accept(&grammar, *it);
    }

    grammar.partial_utf8 = decoded.second;
    if (grammar.stacks.empty()) {
        throw std::runtime_error("Unexpected empty grammar stack after accepting piece: " + piece);
    }
}
```
So first the string will be decoded into:
```console
(gdb) p decoded
$23 = {first = std::vector of length 3, capacity 3 = {32, 123, 0}, second = {value = 123, n_remain = 0}}
```
Now, `decode_utf8` as the following signature:
```c++
static std::pair<std::vector<uint32_t>, llama_partial_utf8> decode_utf8(
        const std::string & src,
        llama_partial_utf8 partial_start) {
```
An UTF-8 encoded string can be 1-4 bytes long. The first part of the pair is
the code points and the second part is the partial utf8 structure. What partial
means in this case is that since we are decoding in chunks we might have gotten
an incomplete utf8 character, like we might have only gotten the two bytes of
an UTF-8 character that is 3 bytes long. So the second part of the pair is
telling us if there are any remaining bytes that we need to decode. We need
this information when decoding the next byte so that we generate valid utf8
characters.
```console
(gdb) p decoded.first
$25 = std::vector of length 3, capacity 3 = {32, 123, 0}
(gdb) p piece
$26 = " {"
```
So the white space character is 32, and the `{` character is 123. The 0 is the
null terminating character.
The code above will then iterate over the code points, skipping the last one
which is the null terminating character. So first 32 will be passed to
`llama_grammar_accept`:
```c++
void llama_grammar_accept(struct llama_grammar * grammar, uint32_t chr) {
    llama_grammar_stacks stacks_new;
    stacks_new.reserve(grammar->stacks.size());

    for (const auto & stack : grammar->stacks) {
        if (stack.empty()) {
            continue;
        }

        auto match = llama_grammar_match_char(stack.back(), chr);
        if (match.first) {
            const llama_grammar_element * pos = match.second;

            // update top of stack to next element, if any
            llama_grammar_stack new_stack(stack.begin(), stack.end() - 1);
            if (!llama_grammar_is_end_of_sequence(pos)) {
                new_stack.push_back(pos);
            }
            llama_grammar_advance_stack(grammar->rules, new_stack, stacks_new);
        }
    }

    grammar->stacks = std::move(stacks_new);
}
```
```console
(gdb) p grammar.stacks.size()
$29 = 1

(gdb) p *stack.front()
$37 = {type = LLAMA_GRETYPE_CHAR, value = 123}
```
So we are first going to try to match the white space.


### stacks
The `llama_grammar` struct has a member `stacks` which is a vector of stacks.
```c++
struct llama_grammar {
    const llama_vocab * vocab;

    const llama_grammar_rules  rules;
          llama_grammar_stacks stacks;
    ...
};

using llama_grammar_stacks = std::vector<llama_grammar_stack>;
using llama_grammar_stack = std::vector<const llama_grammar_element *>;

```
So a grammar can have multiple stacks, and one stack is a vector of grammar
elements:
```console
stasks ------> [stack1]  ---> [element1, element2, element3]
               [stack2]  ---> [element1, element2, element3]


struct llama_grammar_stack = std::vector<const llama_grammar_element*>;  // One possible path
struct llama_grammar_stacks = std::vector<llama_grammar_stack>;         // All possible paths
```
```c++
typedef struct llama_grammar_element {
    enum llama_gretype type;
    uint32_t           value; // Unicode code point or rule ID
} llama_grammar_element;
```
```console
(gdb) p stacks
$1 = std::vector of length 1, capacity 1 = {std::vector of length 1, capacity 1 = {0x55555e707110}}

(gdb) p *stacks[0][0]
$11 = {type = LLAMA_GRETYPE_CHAR, value = 123}
```
This is the next immediate character that it is expecting. So this is already
in the stack. Only the next expected character is pushed onto the stack, not
all possible future characters. When it has seen a '{' then it will push
something else.
```console
root   ::= object
value  ::= object | array | string | number | ("true" | "false" | "null") ws

object ::=
  "{" ws (
            string ":" ws value
    ("," ws string ":" ws value)*
  )? "}" ws

array  ::=
  "[" ws (
            value
    ("," ws value)*
  )? "]" ws

string ::=
  "\"" (
    [^"\\\x7F\x00-\x1F] |
    "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4}) # escapes
  )* "\"" ws

number ::= ("-"? ([0-9] | [1-9] [0-9]{0,15})) ("." [0-9]+)? ([eE] [-+]? [0-9] [1-9]{0,15})? ws

# Optional space: by convention, applied in this grammar after literal chars when allowed
ws ::= | " " | "\n" [ \t]{0,20}
```
So we read this as the root contains an object, ::= means that it is expanded
or defined. And an object must start with a '{', followed an optional whitespace.
Following that we have parenthesis that are used for grouping, and in this case
it is an optional group because of the ? following it. The contents can then be
a string followed by a ':', an optional whitespace, and then a value.
So the stack is just the first character expected and as we can see this is the
'{' character (token 123). 

```c++
void llama_grammar_accept(struct llama_grammar * grammar, uint32_t chr) {
    llama_grammar_stacks stacks_new;
    stacks_new.reserve(grammar->stacks.size());

    for (const auto & stack : grammar->stacks) {
        if (stack.empty()) {
            continue;
        }

        auto match = llama_grammar_match_char(stack.back(), chr);
        if (match.first) {
            const llama_grammar_element * pos = match.second;

            // update top of stack to next element, if any
            llama_grammar_stack new_stack(stack.begin(), stack.end() - 1);
            if (!llama_grammar_is_end_of_sequence(pos)) {
                new_stack.push_back(pos);
            }
            llama_grammar_advance_stack(grammar->rules, new_stack, stacks_new);
        }
    }

    grammar->stacks = std::move(stacks_new);
}
```
Notice that this is calling `llama_grammar_match_char` and passing in the back
of the vector of which is a llama_grammar_element:
```console
(gdb) p chr
$2 = 32
(gdb) p grammar->stacks
$3 = std::vector of length 1, capacity 1 = {std::vector of length 1, capacity 1 = {0x55555e706560}}
(gdb) p grammar->stacks[0]
$4 = std::vector of length 1, capacity 1 = {0x55555e706560}
(gdb) p grammar->stacks[0][0]
$5 = (const llama_grammar_element *) 0x55555e706560
(gdb) p *grammar->stacks[0][0]
$6 = {type = LLAMA_GRETYPE_CHAR, value = 123}
```
```c++
static std::pair<bool, const llama_grammar_element *> llama_grammar_match_char(
        const llama_grammar_element * pos,
        const uint32_t                chr) {
    bool found            = false;
    bool is_positive_char = pos->type == LLAMA_GRETYPE_CHAR || pos->type == LLAMA_GRETYPE_CHAR_ANY;

    GGML_ASSERT(is_positive_char || pos->type == LLAMA_GRETYPE_CHAR_NOT); // NOLINT

    do {
        if (pos[1].type == LLAMA_GRETYPE_CHAR_RNG_UPPER) {
            // inclusive range, e.g. [a-z]
            found = found || (pos->value <= chr && chr <= pos[1].value);
            pos += 2;
        } else if (pos->type == LLAMA_GRETYPE_CHAR_ANY) {
            // Any character matches "."
            found = true;
            pos += 1;
        } else {
            // exact char match, e.g. [a] or "a"
            found = found || pos->value == chr;
            pos += 1;
        }
    } while (pos->type == LLAMA_GRETYPE_CHAR_ALT);

    return std::make_pair(found == is_positive_char, pos);
}
```
There will not be a match because the 32 (space) does not match the 123 ('{'),
so the above will return a pair with false and pos.
```console
(gdb) p match
$8 = {first = false, second = 0x55555e706568}
```
So the grammar expects '{' but it got a space. So the stack will not be updated
and will become empty instead as the grammar is invalid.

First we get the string representation of the token:
```c++
void llama_grammar_accept_impl(struct llama_grammar & grammar, llama_token token) {
    GGML_ASSERT(grammar.vocab != nullptr);

    const auto & piece = grammar.vocab->token_to_piece(token);
```
Now, this might contains a word boundary character depending on the the models
tokenizer. So if we have the trigger word '{' this would actually become " {"
when we get the string representation of the token using token_to_piece. This
means that the first code point to be matched against the current grammar stack
will be 32 (space) and not 123 ('{').

We match the trigger on the token:
```console
(gdb) p grammar.trigger_tokens
$21 = std::vector of length 1, capacity 1 = {426}
$22 = {text = "▁{", score = -167, attr = LLAMA_TOKEN_ATTR_NORMAL}
(gdb) p grammar.vocab->token_to_piece(426)
$23 = " {"
```
I wondering if this is could possibly be a bug and that this matching should be
done without the word boundary character?
I'm not sure if this is a bug or not but making the following change allowed the
example above to pass and did not cause any test failures:
```console
diff --git a/src/llama-grammar.cpp b/src/llama-grammar.cpp
index 9b518d1a..c2e93d2e 100644
--- a/src/llama-grammar.cpp
+++ b/src/llama-grammar.cpp
@@ -1163,7 +1163,17 @@ void llama_grammar_apply_impl(const struct llama_grammar & grammar, llama_token_
 void llama_grammar_accept_impl(struct llama_grammar & grammar, llama_token token) {
     GGML_ASSERT(grammar.vocab != nullptr);

-    const auto & piece = grammar.vocab->token_to_piece(token);
+    std::string piece;
+    piece.resize(piece.capacity());
+    const int len = grammar.vocab->token_to_piece(token, &piece[0], piece.size(), true, true);
+    if (len < 0) {
+        piece.resize(-n_chars);
+        int check = grammar.vocab->token_to_piece(token, &piece[0], piece.size(), true, true);
+        GGML_ASSERT(check == -len);
+    }
+    else {
+        piece.resize(len);
+    }

     if (grammar.awaiting_trigger) {
         if (std::find(grammar.trigger_tokens.begin(), grammar.trigger_tokens.end(), token) != grammar.trigger_tokens.end()) {
```


### Grammar parsing background
Imagine we have rules defined for a grammar like the following:
```
function_call ::= name '(' args ')'
name ::= [a-zA-Z]+
args ::= value | value ',' args
value ::= [0-9]+ | name
```
This means that we have a root element called `function_call` that is defined
as the stuff after `::=` which means expaned to I think. `name` is like a
variable which is defined on the following row. This is called a non-terminal 
because it can be expanded. Then name is defined as a sequence of characters
from a-z or A-Z. The `+` means that it can be repeated one or more times.
The '(' is a terminal because it is a single character and not a rule.

The parsing uses a stack to keep track of the current state of the parser.
Lets take the following example:
```
"foo(18,"
```
So this is intentionally missing the closing parenthesis. The parser will
start out in the following state:
```console
Stack:    [function_call]
Which we know is defined as:  name '(' args ')'
1. Push the rules in reverse order onto the stack:
Push ')'
Push args
Push '('
Push name
Stack:    [name]
          ['(']
          [args]
          [')']

We pop non-terminal rules, and name is currentlty at the top of the stack. So
we pop and expand it:
Pop: name
Push: [a-zA-Z]+
Stack:    [a-zA-Z]+
          ['(']
          [args]
          [')']
```
Then the parser will read the first character 'f':
```
[[a-zA-Z]+ '(' args ')']

'f' matches [a-zA-Z]+
```
And since this pattern is still fullfilled it stays on the stack. Then we parse
'o':
```
[[a-zA-Z]+ '(' args ')']
'o' matches [a-zA-Z]+
```
And the same for the following 'o'.
Next we have '(':
```
[[a-zA-Z]+ '(' args ')']
'(' does not match [a-zA-Z]+, so we pop that rule off the stack.
```

```
['(' args ')']
'(' matches '(' so we pop the stack again and expand the next rule and continue parsing.
[[0-9]+ | [[a-zA-Z]+ ')']
```
Then we have '1' which matches [0-9]+, and the same for '8'. Then we have ','
which also matches. But then we reach the last character and the stack is not
empty:
```
[[0-9]+ | [[a-zA-Z]+ ')']
```
So in this was the grammer knows that the string passed is not a valid function.

This is called pushdown automata which is because future rules can be pushed
onto the stack for later processing. This enabled handling nested structures.

This is called pushdown automata because rules get pushed down onto a stack and
then popped off as they're processed, with new rules being pushed down during
expansion. This stack mechanism enables handling nested structures by keeping
track of what needs to be matched later.


With some backgound out of the way lets look what the implementation in
llama.cpp looks like:
```console
$ gdb --args ./build/bin/llama-cli -m ../llama.cpp/models/Meta-Llama-3.1-8B-Instruct-Q3_K_S.gguf --no-warmup --prompt '"What is LoRA?"' -ngl 40 --grammar-file grammars/json.gbnf -no-cnv -n 40
(gdb) br llama_grammar_init_impl
```
In main.cpp we have:
```c++
    smpl = common_sampler_init(model, sparams);
```
```c++
struct common_sampler * common_sampler_init(const struct llama_model * model, const struct common_params_sampling & params) {
    const llama_vocab * vocab = llama_model_get_vocab(model);
    ...

        grmr = params.grammar_lazy
             ? llama_sampler_init_grammar_lazy(vocab, params.grammar.c_str(), "root",
                                               trigger_words.data(), trigger_words.size(),
                                               params.grammar_trigger_tokens.data(), params.grammar_trigger_tokens.size())
             :      llama_sampler_init_grammar(vocab, params.grammar.c_str(), "root");
```
And this will end up in:
```c++
struct llama_sampler * llama_sampler_init_grammar(
        const struct llama_vocab * vocab,
                      const char * grammar_str,
                      const char * grammar_root) {
    return llama_sampler_init_grammar_impl(vocab, grammar_str, grammar_root, /* lazy= */ false, nullptr, 0, nullptr, 0);
}

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
```
Which leads us to (in llama-grammar.cpp):
```c++
struct llama_grammar * llama_grammar_init_impl(
        const struct llama_vocab * vocab,
                      const char * grammar_str,
                      const char * grammar_root,
                              bool lazy,
                     const char ** trigger_words,
                            size_t num_trigger_words,
               const llama_token * trigger_tokens,
                            size_t num_trigger_tokens) {
    llama_grammar_parser parser;

    // if there is a grammar, parse it
    if (!parser.parse(grammar_str)) {
        return nullptr;
    }
    ...
```

```c++
bool llama_grammar_parser::parse(const char * src) {
    try {
        const char * pos = parse_space(src, true);
        while (*pos) {
            pos = parse_rule(pos);
        }
```
The call to `parse_space` is just to skip any leadning whitespaces or comments
and after that call pos will point to the first character of the rule.
Now, lets take a look at `parse_rule`, and recall that pos is:
```console
(gdb) p pos
$5 = 0x55555610e930 "root   ::= object\nvalue  ::= object | array | string | number | (\"true\" | \"false\" | \"null\") ws\n\nobject ::=\n  \"{\" ws (\n", ' ' <repeats 12 times>, "string \":\" ws value\n    (\",\" ws string \":\" ws value)*\n  )? \"}\" ws\n\narr"...
```

```c++
const char * llama_grammar_parser::parse_rule(const char * src) {
    const char * name_end = parse_name(src);
    const char * pos      = parse_space(name_end, false);
    size_t       name_len = name_end - src;
    uint32_t     rule_id  = get_symbol_id(src, name_len);
    const std::string name(src, name_len);

    if (!(pos[0] == ':' && pos[1] == ':' && pos[2] == '=')) {
        throw std::runtime_error(std::string("expecting ::= at ") + pos);
    }
    pos = parse_space(pos + 3, true);

    pos = parse_alternates(pos, name, rule_id, false);

    if (*pos == '\r') {
        pos += pos[1] == '\n' ? 2 : 1;
    } else if (*pos == '\n') {
        pos++;
    } else if (*pos) {
        throw std::runtime_error(std::string("expecting newline or end at ") + pos);
    }
    return parse_space(pos, true);
}
```
So first the name of the rule will be parsed:
```c++
static const char * parse_name(const char * src) {
    const char * pos = src;
    while (is_word_char(*pos)) {
        pos++;
    }
    if (pos == src) {
        throw std::runtime_error(std::string("expecting name at ") + src);
    }
    return pos;
}
```
This is done by iterating over the characters until we reach a character that
is not a word character. And notice the pos will point to the first character
after the name. A
Then we have the `parse_space` call which we've already seen. Then the length
of the rule name will be calculated.
Following that we have the `get_symbol_id` call which is passing in the name
length, and the original source string:
```c++
(gdb) p src
$8 = 0x55555610e930 "root   ::= object\nvalue  ::= object | array | string | number | (\"true\" | \"false\" | \"null\") ws\n\nobject ::=\n  \"{\" ws (\n", ' ' <repeats 12 times>, "string \":\" ws value\n    (\",\" ws string \":\" ws value)*\n  )? \"}\" ws\n\narr"...
(gdb) p name_len
$7 = 4
```
This will use the `symbol_ids` std::map (not std::unordered_map which I wonder
what the reason was for chosing std::map instead of std::unordered) to get the
a unique id for the symbol. 
```c++
uint32_t llama_grammar_parser::get_symbol_id(const char * src, size_t len) {
    uint32_t next_id = static_cast<uint32_t>(symbol_ids.size());
    auto result = symbol_ids.emplace(std::string(src, len), next_id);
    return result.first->second;
}
```
This is using the size of the map as the next id, and then it is inserting the
into the map. The `emplace` function will return a pair where the first is the
entry inserted (if it was) which is also a pair. And the second value of the pair
is a  bool indicating if the element was inserted or not:
```console
(gdb) p result
$10 = {first = {first = "root", second = 0}, second = true}
```
Following that we have:
```c++
    const std::string name(src, name_len);

    if (!(pos[0] == ':' && pos[1] == ':' && pos[2] == '=')) {
        throw std::runtime_error(std::string("expecting ::= at ") + pos);
    }
    pos = parse_space(pos + 3, true);

    pos = parse_alternates(pos, name, rule_id, false);
```
This is just checking for the existence of the `::=` and then skipping any
additional whitespaces or comments.
Lets now look at `parse_alternates`:
```console
(gdb) p rule_id
$13 = 0
(gdb) p name
$14 = "root"
(gdb) p pos
$15 = 0x55555610e93b "object\nvalue  ::= object | array | string | number | (\"true\" | \"false\" | \"null\") ws\n\nobject ::=\n  \"{\" ws (\n", ' ' <repeats 12 times>, "string \":\" ws value\n    (\",\" ws string \":\" ws value)*\n  )? \"}\" ws\n\narray  ::=\n  \""...
```

```c++
const char * llama_grammar_parser::parse_alternates(
        const char        * src,
        const std::string & rule_name,
        uint32_t            rule_id,
        bool                is_nested) {
    llama_grammar_rule rule;
    const char * pos = parse_sequence(src, rule_name, rule, is_nested);
    while (*pos == '|') {
        rule.push_back({LLAMA_GRETYPE_ALT, 0});
        pos = parse_space(pos + 1, true);
        pos = parse_sequence(pos, rule_name, rule, is_nested);
    }
    rule.push_back({LLAMA_GRETYPE_END, 0});
    add_rule(rule_id, rule);
    return pos;
}
```
So this is creating a `llama_grammar_rule` which is a vector of
`llama_grammar_element`:
```console
(gdb) ptype rule
type = std::vector<llama_grammar_element>
```
And passing this to `parse_sequence`:
```c++
const char * llama_grammar_parser::parse_sequence(
        const char         * src,
        const std::string  & rule_name,
        llama_grammar_rule & rule,
        bool               is_nested) {
    size_t last_sym_start = rule.size();
    const char * pos = src;

    auto handle_repetitions = [&](int min_times, int max_times) {
        ...
    };
```
Repitions are things like `*`, `+`, `?` and {m,n}.

```c++
    while (*pos) {
        if (*pos == '"') { // literal string
            ...
        } else if (is_word_char(*pos)) { // rule reference
            const char * name_end    = parse_name(pos);
            uint32_t ref_rule_id = get_symbol_id(pos, name_end - pos);
            pos = parse_space(name_end, is_nested);
            last_sym_start = rule.size();
            rule.push_back({LLAMA_GRETYPE_RULE_REF, ref_rule_id});
        } else if (*pos == '(') { // grouping
        ... 
```
So we are seeing `parse_name` again which so this is parsing the name next
part of the rule which is `object` which is a non-terminal.
And then we are calling `get_symbol_id` which will return the id for the rule:
```console
(gdb) p result
$25 = {first = {first = "object", second = 1}, second = true}
```
Following that a rule is pushed onto the stack and notice that this is of type
LLAMA_GRETYPE_RULE_REF. This is the type for non-terminals.
```console
(gdb) p rule
$31 = std::vector of length 1, capacity 1 = {{type = LLAMA_GRETYPE_RULE_REF, value = 1}}
```
After the whole rules file has been parse we will return back in
`llama_grammar_init_impl`:

```c++
    // if there is a grammar, parse it
    if (!parser.parse(grammar_str)) {
        return nullptr;
    }

    // will be empty (default) if there are parse errors
    if (parser.rules.empty()) {
        fprintf(stderr, "%s: failed to parse grammar\n", __func__);
        return nullptr;
    }

    // Ensure that there is a "root" node.
    if (parser.symbol_ids.find("root") == parser.symbol_ids.end()) {
        fprintf(stderr, "%s: grammar does not contain a 'root' symbol\n", __func__);
        return nullptr;
    }

    std::vector<const llama_grammar_element *> grammar_rules(parser.c_rules());
```
What is `c_rules()`?
This is creating a vector of pointers to the rules:
```c++
llama_grammar_stack llama_grammar_parser::c_rules() const {
    llama_grammar_stack ret;
    ret.reserve(rules.size());
    for (const auto & rule : rules) {
        ret.push_back(rule.data());
    }
    return ret;
}
```
Next the is a check for left recursion which is when we have a situation like
the following:
```c
expr ::= expr "+" number  | number
```
In this case to expand `expr` it will need to expand `expr "+" number` and that
means it has to expand `expr` again. This will lead to an infinite loop.
```c++
    // Check for left recursion
    std::vector<bool> rules_visited(n_rules);
    std::vector<bool> rules_in_progress(n_rules);
    std::vector<bool> rules_may_be_empty(n_rules);
    for (size_t i = 0; i < n_rules; i++) {
        if (rules_visited[i]) {
            continue;
        }
        if (llama_grammar_detect_left_recursion(vec_rules, i, &rules_visited, &rules_in_progress, &rules_may_be_empty)) {
            LLAMA_LOG_ERROR("unsupported grammar, left recursion detected for nonterminal at index %zu", i);
            return nullptr;
        }
    }
