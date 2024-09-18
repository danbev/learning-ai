### Llama.cpp Sampling API
This document is about the sampling in llama.cpp. I've written about sampling
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
the index. Then we get the size of the vocabulary and create a vector of of
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
