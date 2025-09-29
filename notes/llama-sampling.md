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
