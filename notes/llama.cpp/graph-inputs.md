### Graph input
There is an interface type in `llama-graph.h`:
```c++
class llm_graph_input_i {
public:
    virtual ~llm_graph_input_i() = default;

    virtual void set_input(const llama_ubatch * ubatch) = 0;

    virtual bool can_reuse(const llm_graph_params & params) {
        GGML_UNUSED(params);
        return false;
    }
};
```
There are a number of concrete inputs defined
* llm_graph_input_embd 
Standard transformer input, either tokens ids or pre-computed token embeddings

* llm_graph_input_pos 
Positional information for each token in the sequence.

* llm_graph_input_attn_temp
Temperature tuning for attention, used by Llama 4.

* llm_graph_input_pos_bucket 
* llm_graph_input_pos_bucket_kv 
Relative position buckets for attention, used by T5, PaLM.

* llm_graph_input_out_ids           (pooling)
For extracting specific tokens (not all)
* llm_graph_input_mean              (pooling)
* llm_graph_input_cls               (pooling)

* lLm_graph_input_rs                (recurrent state)

* llm_graph_input_attn_no_cache
* llm_graph_input_attn_kv_unified 
* llm_graph_input_attn_kv_unified_iswa 

* llm_graph_input_attn_cross 
For cross attention between encoder and decoder.

* llm_graph_input_mem_hybrid 

So lets take a look at one of these and see where they get created. For example,
`llm_graph_input_attn_no_cache` is created and returned by the function
`llm_graph_context::build_attn_inp_no_cache()`.
```c++
llm_graph_input_attn_no_cache * llm_graph_context::build_attn_inp_no_cache() const {
    auto inp = std::make_unique<llm_graph_input_attn_no_cache>(hparams, cparams);

    // note: there is no KV cache, so the number of KV values is equal to the number of tokens in the batch
    inp->kq_mask = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_tokens, GGML_PAD(n_tokens, GGML_KQ_MASK_PAD), 1, 1);
    ggml_set_input(inp->kq_mask);

    inp->kq_mask_cnv = cparams.flash_attn ? ggml_cast(ctx0, inp->kq_mask, GGML_TYPE_F16) : inp->kq_mask;

    return (llm_graph_input_attn_no_cache *) res->add_input(std::move(inp));
}
```
So on the first line the constructor will be called when the unique pointer is
created.
```console
(gdb) p this
$3 = (const llm_graph_context * const) 0x55555593e390
(gdb) f
#2  0x00007ffff7c51a9b in llm_graph_context::build_attn_inp_no_cache (this=0x55555593e390)
    at /home/danbev/work/ai/llama.cpp-tiny-gemma/src/llama-graph.cpp:1296
1296	    auto inp = std::make_unique<llm_graph_input_attn_no_cache>(hparams, cparams);

(gdb) p res
$2 = (llm_graph_result *) 0x5555559592c0

(gdb) ptype res
type = class llm_graph_result {
  public:
    ggml_tensor *t_tokens;
    ggml_tensor *t_logits;
    ggml_tensor *t_embd;
    ggml_tensor *t_embd_pooled;
    std::vector<std::unique_ptr<llm_graph_input_i>> inputs;
    ggml_context_ptr ctx_compute;
    std::vector<unsigned char> buf_compute_meta;
    ggml_cgraph *gf;
    int64_t max_nodes;
  private:
    llm_graph_params params;
    int debug;

  public:
    llm_graph_result(int64_t);
    ~llm_graph_result(void);
    ggml_tensor * get_tokens(void) const;
    ggml_tensor * get_logits(void) const;
    ggml_tensor * get_embd(void) const;
    ggml_tensor * get_embd_pooled(void) const;
    ggml_cgraph * get_gf(void) const;
    ggml_context * get_ctx(void) const;
    int64_t get_max_nodes(void) const;
    void reset(void);
    void set_inputs(const llama_ubatch *);
    bool can_reuse(const llm_graph_params &);
    llm_graph_input_i * add_input(llm_graph_input_ptr);
    void set_params(const llm_graph_params &);
} *
```
At first this might sound a bit strange that a result type has inputs in addition
to output, at least this was this case for me. But if we look this we can see
the there is a `can_resuse` method on this interface. So it is possible that in
stead of rebuilding the computation graph for every inference call the graph can
be reused. So we could use the same model arch and batch configuration but for
different input tokens/embeddings. So this type contains the inputs, the compute
graph, and the results of the computation.

So if we have an embedding model we might have both `llm_graph_input_embd` and
`llm_graph_input_pos`:
```console
(gdb) p res.inputs
$9 = std::vector of length 2, capacity 2 = {std::unique_ptr<llm_graph_input_i> = {get() = 0x5555559af0d0}, 
  std::unique_ptr<llm_graph_input_i> = {get() = 0x5555559af140}}
(gdb) p res.inputs.size()
$10 = 2

(gdb) p *res.inputs[0]
$16 = {_vptr.llm_graph_input_i = 0x7ffff7ef72c8 <vtable for llm_graph_input_embd+16>}
(gdb) p *res.inputs[1]
$17 = {_vptr.llm_graph_input_i = 0x7ffff7ef7298 <vtable for llm_graph_input_pos+16>}
```

The input are created in `llama-model.cpp` and the `llm_build_xxx` functions, 
for example for
```c++
struct llm_build_gemma3_iswa : public llm_graph_context {
    llm_build_gemma3_iswa(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
        const int64_t n_embd_head = hparams.n_embd_head_k;

        ggml_tensor * cur;
        ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // important: do not normalize weights for raw embeddings input (i.e. encoded image emdeddings)
        if (ubatch.token) {
            inpL = ggml_scale(ctx0, inpL, sqrtf(n_embd));
            cb(inpL, "inp_scaled", -1);
        }

        // inp_pos - contains the positions
        ggml_tensor * inp_pos = build_inp_pos();

        // TODO: is causal == true correct? might need some changes
        auto * inp_attn = build_attn_inp_kv_unified_iswa();
```

And later when `process_ubatch` in `llama-context.cpp` is called, the inputs
```c++
llm_graph_result * llama_context::process_ubatch(const llama_ubatch & ubatch,
    llm_graph_type gtype,
    llama_memory_context_i * mctx,
    ggml_status & ret) {
    ...

        res->set_inputs(&ubatch);

}
```
This will call into `llama-graph.cpp`:
```c++
void llm_graph_result::set_inputs(const llama_ubatch * ubatch) {
    for (auto & input : inputs) {
        input->set_input(ubatch);
    }
}
```
And this will first call `llama_graph_input_pos` and then
`llama_graph_input_embd` in this case.
