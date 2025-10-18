## IBM Granite Model
This is a hybrid Mamba2 and transformer model that uses MOE. I'm using this
as an example to better understand how Mamba2 and Hybrid models are implemented
in llama.cpp.

### granite-4.0-h-tiny-Q4_0.gguf
So this is the model that I'm looking at:
```console
(venv) $ gguf-dump ../llama.cpp/models/granite-4.0-h-tiny-Q4_0.gguf
INFO:gguf-dump:* Loading: ../llama.cpp/models/granite-4.0-h-tiny-Q4_0.gguf
* File is LITTLE endian, script is running on a LITTLE endian host.
* Dumping 50 key/value pair(s)
      1: UINT32     |        1 | GGUF.version = 3
      2: UINT64     |        1 | GGUF.tensor_count = 666
      3: UINT64     |        1 | GGUF.kv_count = 47
      4: STRING     |        1 | general.architecture = 'granitehybrid'
      5: STRING     |        1 | general.type = 'model'
      6: STRING     |        1 | general.name = 'Granite-4.0-H-Tiny'
      7: STRING     |        1 | general.basename = 'Granite-4.0-H-Tiny'
      8: STRING     |        1 | general.quantized_by = 'Unsloth'
      9: STRING     |        1 | general.size_label = '64x994M'
     10: STRING     |        1 | general.repo_url = 'https://huggingface.co/unsloth'
     ...
     13: UINT32     |        1 | granitehybrid.embedding_length = 1536
     14: UINT32     |        1 | granitehybrid.feed_forward_length = 512
     15: UINT32     |        1 | granitehybrid.attention.head_count = 12
     16: [INT32]    |       40 | granitehybrid.attention.head_count_kv = [0, 0, 0, 0, 0, 4, ...]
     ...
     21: UINT32     |        1 | granitehybrid.vocab_size = 100352
     ...
     27: UINT32     |        1 | granitehybrid.expert_shared_feed_forward_length = 1024
     28: UINT32     |        1 | granitehybrid.ssm.conv_kernel = 4
     29: UINT32     |        1 | granitehybrid.ssm.state_size = 128
     30: UINT32     |        1 | granitehybrid.ssm.group_count = 1
     31: UINT32     |        1 | granitehybrid.ssm.inner_size = 3072
     32: UINT32     |        1 | granitehybrid.ssm.time_step_rank = 48
     33: BOOL       |        1 | granitehybrid.rope.scaling.finetuned = False
```
Now, let look at what one of the layers look like:
```console
      3:       1536 |  1536,     1,     1,     1 | F32     | blk.0.attn_norm.weight
      4:   50331648 |   512,  1536,    64,     1 | Q4_1    | blk.0.ffn_down_exps.weight
      5:    1572864 |  1024,  1536,     1,     1 | Q6_K    | blk.0.ffn_down_shexp.weight
      6:   50331648 |  1536,   512,    64,     1 | Q4_0    | blk.0.ffn_gate_exps.weight
      7:      98304 |  1536,    64,     1,     1 | F32     | blk.0.ffn_gate_inp.weight
      8:    1572864 |  1536,  1024,     1,     1 | Q5_K    | blk.0.ffn_gate_shexp.weight
      9:       1536 |  1536,     1,     1,     1 | F32     | blk.0.ffn_norm.weight
     10:   50331648 |  1536,   512,    64,     1 | Q4_0    | blk.0.ffn_up_exps.weight
     11:    1572864 |  1536,  1024,     1,     1 | Q5_K    | blk.0.ffn_up_shexp.weight
     12:         48 |     1,    48,     1,     1 | F32     | blk.0.ssm_a
     13:       3328 |  3328,     1,     1,     1 | F32     | blk.0.ssm_conv1d.bias
     14:      13312 |     4,  3328,     1,     1 | F32     | blk.0.ssm_conv1d.weight
     15:         48 |     1,    48,     1,     1 | F32     | blk.0.ssm_d
     16:         48 |    48,     1,     1,     1 | F32     | blk.0.ssm_dt.bias
     17:    9904128 |  1536,  6448,     1,     1 | Q4_0    | blk.0.ssm_in.weight
     18:       3072 |  3072,     1,     1,     1 | F32     | blk.0.ssm_norm.weight
     19:    4718592 |  3072,  1536,     1,     1 | Q4_0    | blk.0.ssm_out.weight
```

```console
gdb --args \
    ./build/bin/llama-cli \
    -m ../llama.cpp/models/granite-4.0-h-tiny-Q4_0.gguf \
    --no-warmup \
    --prompt '"What is the Capital of France?"' \
    -n 10 -no-cnv
```

```c++
    llm_build_granite_hybrid(
                 const llama_model & model,
            const llm_graph_params & params) :
        llm_graph_context_mamba(params) {

        const int64_t n_embd_head = hparams.n_embd_head_v;
```

To actually see the infrerence what we are running we have to continue a few
times to skip the graph reservations that happend in the llama-context constructor.
But after this we have the followign
```console
"What is the Capital of France?"
Thread 1 "llama-cli" hit Breakpoint 4, llm_build_granite_hybrid::llm_build_granite_hybrid (this=0x55555766c950, model=..., 
    params=...) at /home/danbev/work/ai/llama.cpp-nemo/src/llama-model.cpp:16184
16184	        inpL = build_inp_embd(model.tok_embd);
(gdb) p ubatch
$38 = (const llama_ubatch &) @0x7fffffffad60: {b_equal_seqs = 1, n_tokens = 7, n_seq_tokens = 7, n_seqs = 1, n_seqs_unq = 1, 
  token = 0x555555ed0f80, embd = 0x0, pos = 0x555555db7f30, n_seq_id = 0x555555db6d60, seq_id = 0x555555ed73e0, 
  seq_id_unq = 0x555555ed1040, seq_idx = 0x555555e08510, output = 0x555555db7060 "", 
  data = std::shared_ptr<llama_ubatch::data_t> (use count 5, weak count 0) = {get() = 0x555555f48ad0}}

(gdb) p model.vocab.pimpl->id_to_token[0]
$52 = {text = "!", score = 0, attr = LLAMA_TOKEN_ATTR_NORMAL}
(gdb) p model.vocab.pimpl->id_to_token[ubatch.token[0]]
$53 = {text = "\"What", score = 0, attr = LLAMA_TOKEN_ATTR_NORMAL}
(gdb) p model.vocab.pimpl->id_to_token[ubatch.token[1]]
$54 = {text = "Ġis", score = 0, attr = LLAMA_TOKEN_ATTR_NORMAL}
(gdb) p model.vocab.pimpl->id_to_token[ubatch.token[2]]
$55 = {text = "Ġthe", score = 0, attr = LLAMA_TOKEN_ATTR_NORMAL}
(gdb) p model.vocab.pimpl->id_to_token[ubatch.token[3]]
$56 = {text = "ĠCapital", score = 0, attr = LLAMA_TOKEN_ATTR_NORMAL}
(gdb) p model.vocab.pimpl->id_to_token[ubatch.token[4]]
$57 = {text = "Ġof", score = 0, attr = LLAMA_TOKEN_ATTR_NORMAL}
(gdb) p model.vocab.pimpl->id_to_token[ubatch.token[5]]
$58 = {text = "ĠFrance", score = 0, attr = LLAMA_TOKEN_ATTR_NORMAL}
(gdb) p model.vocab.pimpl->id_to_token[ubatch.token[6]]
$59 = {text = "?\"", score = 0, attr = LLAMA_TOKEN_ATTR_NORMAL}
```
So we can see that the tokens are what we expect them to be. And we will look
up the embeddings for these tokens simliar to other models:
```c++
        ggml_tensor * cur;
        ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);
```
This model has a vocab of 100352 and a embedding dimension of 1536:
```console
(gdb) p *model.tok_embd
$13 = {type = GGML_TYPE_Q6_K, buffer = 0x5555579904f0, ne = {1536, 100352, 1, 1}, nb = {210, 1260, 126443520, 126443520}, 
  op = GGML_OP_NONE, op_params = {0 <repeats 16 times>}, flags = 0, src = {0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0}, 
  view_src = 0x0, view_offs = 0, data = 0x7ffec516dba0, name = "token_embd.weight", '\000' <repeats 46 times>, extra = 0x0, 
  padding = "\000\000\000\000\000\000\000"}
```
So `inpL` will be a tensor of shape [1536, 7] after this call.


Next we have the memory for the hybrid model which will have both a kv cache
and the ssm state.
```c++
        auto * inp = build_inp_mem_hybrid();
```
TODO: look closer at the hybrid memory, but for now I want to focus on Mamba2.

This model has 40 layers:
```console
        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor * inpSA = inpL;
```
I think SA here means self-attention. First we have a normalization layer:
```c++
            // norm
            cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);
```
Next we have either a recurrent layer (Mamba2) or an attention layer:
```c++
            if (hparams.is_recurrent(il)) {
                // ssm layer //
                cur = build_mamba2_layer(inp->get_recr(), cur, model, ubatch, il);
            } else {
                // attention layer //
                cur = build_attention_layer(
                    cur, inp_pos, inp->get_attn(), model,
                    n_embd_head, il);
            }
```
```console
(gdb) p recurrent_layer_arr
$64 = {_M_elems = {
    true,
    true,
    true,
    true,
    true,
    false,
    true,
    true,
    true,
    true,
    true,
    true,
    true,
    true,
    true,
    false,
    ...
```
So this layer is a recurrent layer. The following builds a single Mamba2 layer:
```c++
    ggml_tensor * build_mamba2_layer(
        llm_graph_input_rs * inp,
               ggml_tensor * cur,
         const llama_model & model,
        const llama_ubatch & ubatch,
                       int   il) const {

        const auto * mctx_cur = inp->mctx;

        const auto kv_head = mctx_cur->get_head();

        const int64_t d_conv  = hparams.ssm_d_conv;
        const int64_t d_inner = hparams.ssm_d_inner;
        const int64_t d_state = hparams.ssm_d_state;
        const int64_t n_head  = hparams.ssm_dt_rank;
        const int64_t head_dim = d_inner / n_head;
        const int64_t n_group = hparams.ssm_n_group;
        const int64_t n_seqs  = ubatch.n_seqs;

        const int64_t n_seq_tokens = ubatch.n_seq_tokens;

```

```console
(gdb) p d_conv
$69 = 4
(gdb) p d_state
$70 = 128
(gdb) p n_head
$71 = 48
(gdb) p head_dim
$72 = 64
(gdb) p n_group
$73 = 1
(gdb) p n_seqs
$74 = 1
(gdb) p n_seq_tokens 
$75 = 7
```

```
Next we have:
```c++
        ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
```
Lets start by looking at conv_states_all:
```console
(gdb) p conv_states_all
(gdb) p *conv_states_all
$78 = {type = GGML_TYPE_F32, buffer = 0x555555dbd000, ne = {9984, 1, 1, 1}, nb = {4, 39936, 39936, 39936}, op = GGML_OP_NONE,
  op_params = {0 <repeats 16 times>}, flags = 0, src = {0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0}, view_src = 0x0,
  view_offs = 0, data = 0x7ffdecdbb040, name = "cache_r_l0", '\000' <repeats 53 times>, extra = 0x0,
  padding = "\000\000\000\000\000\000\000"}
````
The number of previous timesteps that are stored is d_conv - 1:
```
dconv - 1 = previous timesteps to store
4     - 1 = 3
```
The dimension of each of these states is:
```
dimension = d_inner + 2 * n_group * d_state
d_inner = n_head * head_dim
d_inner = 48 * 64 = 3072
2 * n_group * d_state = 2 * 1 * 128 = 256

dimension = 3072 + 256 = 3328
(d_conv - 1) * dimension = 3 * 3328 = 9984
```
Now, from above we saw that the embedding dimension is 1536, and if my understanding
of the usage of this information is correct it is supposed to enable the current
token to influence or give some local context using the last three tokens. So
for this to add up we have to take into account that the token embedding will
be projected xBC vector of dimension 3328. The convolution state stores these
projected xBC vectors from previous timesteps, not the original embeddings.

Next, we will pass the conv_states_all to build_rs:
```c++
        ggml_tensor * conv = build_rs(inp, conv_states_all, hparams.n_embd_r(), n_seqs);
```
```console
(gdb) p hparams.n_embd_r()
$80 = 9984
```
With that lets look at build_rs:
```c++
ggml_tensor * llm_graph_context::build_rs(
        llm_graph_input_rs * inp,
        ggml_tensor * s,   // conv_states_all
            int32_t   state_size,
            int32_t   n_seqs,
        const llm_graph_get_rows_fn & get_state_rows) const {
    const auto * kv_state = inp->mctx;

    return build_rs(s, inp->s_copy_main, inp->s_copy_extra, state_size, n_seqs,
                    kv_state->get_n_rs(), kv_state->get_head(), kv_state->get_size(), kv_state->get_rs_z(),
                    get_state_rows);
}
```
So notice that we are passing in the conv_states_all as `s` here which is the
which contains state for each sequence (we only have one sequence in this case):
```c++
ggml_tensor * llm_graph_context::build_rs(
        ggml_tensor * s,                  // conv_states_all
        ggml_tensor * state_copy_main,
        ggml_tensor * state_copy_extra,
            int32_t   state_size,
            int32_t   n_seqs,
           uint32_t   n_rs,
           uint32_t   rs_head,
           uint32_t   rs_size,
            int32_t   rs_zero,
        const llm_graph_get_rows_fn & get_state_rows) const {

    ggml_tensor * states = ggml_reshape_2d(ctx0, s, state_size, rs_size);

    // Clear a single state which will then be copied to the other cleared states.
    // Note that this is a no-op when the view is zero-sized.
    ggml_tensor * state_zero = ggml_view_1d(ctx0, states, state_size*(rs_zero >= 0), rs_zero*states->nb[1]*(rs_zero >= 0));
    ggml_build_forward_expand(gf, ggml_scale_inplace(ctx0, state_zero, 0));
    // Interesting way to zero out the state, using a scale_inplace with 0. I must remember this.

    // copy states
    // {state_size, rs_size} -> {state_size, n_seqs}
    ggml_tensor * output_states = get_state_rows(ctx0, states, state_copy_main);
    // Uses indices in state_copy_main to select which rows to copy for the
    // three previous states.
    ggml_build_forward_expand(gf, output_states);

    // copy extra states which won't be changed further (between n_seqs and n_rs)
    ggml_tensor * states_extra = ggml_get_rows(ctx0, states, state_copy_extra);
    // This handles states that are part of the current request but won't be
    // actively modified. This is for multi-sequence handling where some sequences
    // might be in the batch but aren't being processed.

    ggml_build_forward_expand(gf,
        ggml_cpy(ctx0, states_extra,
            ggml_view_1d(ctx0, s, state_size*(n_rs - n_seqs), (rs_head + n_seqs)*state_size*ggml_element_size(s))));

    // and we return the selected previous states for the current sequence.
    return output_states;
}
```
Imagine we had 3 conversations active, and cache capacity for 8 sequences:
```
Cache slots: [0] [1] [2] [3] [4] [5] [6] [7]
              A   B   C   -   -   -   -   -
```
Current batch processes conversations A and C:
```
n_seqs = 2
state_copy_main = [0, 2] (grab slots 0 and 2)
rs_head = 3 (next write goes to slot 3)
If a new conversation starts, rs_zero = 3 (zero slot 3)
```
So back in build_mamba2_layer we are going to reshape the output from build_rs
which returns the previous states for the current sequences:
```c++
        conv = ggml_reshape_3d(ctx0, conv, d_conv - 1, d_inner + 2*n_group*d_state, n_seqs);
```
And we can see a little more clearly what we talked about before about this size
earlier:
```console
(gdb) p d_conv -1
$94 = 3

(gdb) p d_inner + 2 * n_group * d_state
$95 = 3328

(gdb) p conv->ne
$96 = {3, 3328, 1, 1}
```
And cur, which is just the input token embeddings is reshaped to a 3d tensor:
```console
(gdb) p *cur
$100 = {type = GGML_TYPE_F32, buffer = 0x0, ne = {1536, 7, 1, 1}, nb = {4, 6144, 43008, 43008}, op = GGML_OP_MUL, op_params = {
    0 <repeats 16 times>}, flags = 0, src = {0x55555628e5d0, 0x55555799fc80, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0}, 
  view_src = 0x0, view_offs = 0, data = 0x0, name = "attn_norm-0", '\000' <repeats 52 times>, extra = 0x0, 
  padding = "\000\000\000\000\000\000\000"}
```
```c++
        cur = ggml_reshape_3d(ctx0, cur, cur->ne[0], n_seq_tokens, n_seqs);
```
In our case this will not do anything as we have only one sequence.
```c++
        ggml_tensor * zxBCdt = build_lora_mm(model.layers[il].ssm_in, cur);
```
```console
(gdb) p *model.layers[il].ssm_in
$103 = {type = GGML_TYPE_Q4_0, buffer = 0x555557a25f10, ne = {1536, 6448, 1, 1}, nb = {18, 864, 5571072, 5571072}, 
  op = GGML_OP_NONE, op_params = {0 <repeats 16 times>}, flags = 0, src = {0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0}, 
  view_src = 0x0, view_offs = 0, data = 0x7ffdf051b040, name = "blk.0.ssm_in.weight", '\000' <repeats 44 times>, 
  extra = 0x7ffff777b1e8 <ggml_repack_get_optimal_repack_type(ggml_tensor const*)::q4_0_8x8_q8_0>, 
  padding = "\000\000\000\000\000\000\000"}
```
Now, this is an interesting operation, this is a matrix multiplication that is
expanding the input token embedding into the space required for Mamba2.
So this is one large matrix multiplication for efficiency and then we can create
views into this for each of the parts. Notice that we used cur for with this
matrix multiplication so this is how the input tokens are used in the z, B, C,
and dt operations/values.

This projects each of the 7 tokens from 1536 dimensions to 6448 dimensions.
The 6448 dimensions are made up of 3 parts:
```
ggml_tensor * z   = ...;    // head_dim * n_head = 64 * 48 = 3072
ggml_tensor * xBC = ...;    // d_inner + 2*n_group*d_state = 3328
ggml_tensor * dt  = ...;    // n_head                      = 48

                                                     Total = 6448
```
z is the gating signal that will be used with SwiGLU.
```c++
        ggml_tensor * z = ggml_view_4d(ctx0, zxBCdt,
            head_dim,
            n_head,
            n_seq_tokens,
            n_seqs,
            head_dim*zxBCdt->nb[0], zxBCdt->nb[1], zxBCdt->nb[2],
            0);
```
```console
(gdb) p *z
$106 = {type = GGML_TYPE_F32, buffer = 0x0, ne = {64, 48, 7, 1}, nb = {4, 256, 25792, 180544}, op = GGML_OP_VIEW, op_params = {
    0 <repeats 16 times>}, flags = 0, src = {0x55555628f5a0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0},
  view_src = 0x55555628f5a0, view_offs = 0, data = 0x0, name = " (view)", '\000' <repeats 56 times>, extra = 0x0,
  padding = "\000\000\000\000\000\000\000"}
```
xBC will be used for the convolution and what will later be stored in conv_states.
```c++
        ggml_tensor * xBC = ggml_view_3d(ctx0, zxBCdt,
            d_inner + 2*n_group*d_state,
            n_seq_tokens,
            n_seqs,
            zxBCdt->nb[1], zxBCdt->nb[2],
            d_inner*ggml_element_size(zxBCdt));
```
dt is the delta which controls the descretization of the SSM and there is one
value per head:
```c++
        ggml_tensor * dt = ggml_view_3d(ctx0, zxBCdt,
            n_head,
            n_seq_tokens,
            n_seqs,
            zxBCdt->nb[1],
            zxBCdt->nb[2],
            (2*d_inner + 2*n_group*d_state)*ggml_element_size(zxBCdt));
```

Then we have the convolution:
```c++
        // conv
        {
            ggml_tensor * conv_x = ggml_concat(ctx0, conv, ggml_transpose(ctx0, xBC), 0);
```
So this is taking the 3 previous timesteps from the state cache:
```console
(gdb) p *conv
$115 = {type = GGML_TYPE_F32, buffer = 0x0, ne = {3, 3328, 1, 1}, nb = {4, 12, 39936, 39936}, op = GGML_OP_RESHAPE, op_params = {
    0 <repeats 16 times>}, flags = 0, src = {0x55555628ed00, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0},
  view_src = 0x55555628ed00, view_offs = 0, data = 0x0, name = "node_4 (reshaped)", '\000' <repeats 46 times>, extra = 0x0,
  padding = "\000\000\000\000\000\000\000"}
```
And the current 7 tokens projected into xBC transposed:
```console
(gdb) p *ggml_transpose(ctx0, xBC)
$113 = {type = GGML_TYPE_F32, buffer = 0x0, ne = {7, 3328, 1, 1}, nb = {25792, 4, 180544, 180544}, op = GGML_OP_TRANSPOSE,
  op_params = {0 <repeats 16 times>}, flags = 0, src = {0x55555628f880, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0},
  view_src = 0x55555628f5a0, view_offs = 12288, data = 0x0, name = " (view) (transposed)", '\000' <repeats 43 times>, extra = 0x0,
  padding = "\000\000\000\000\000\000\000"}
```
So this will give us 10 timesteps:
```console
(gdb) p *conv_x
$114 = {type = GGML_TYPE_F32, buffer = 0x0, ne = {10, 3328, 1, 1}, nb = {4, 40, 133120, 133120}, op = GGML_OP_CONCAT, op_params = {
    0 <repeats 16 times>}, flags = 0, src = {0x55555628f2c0, 0x55555628fcd0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0},
  view_src = 0x0, view_offs = 0, data = 0x0, name = '\000' <repeats 63 times>, extra = 0x0, padding = "\000\000\000\000\000\000\000"}
```
Next we create a view of size 3 and notice the offset (the last argument is
n_seq_tokens which is 7 (and we have 10 elements in conv_x), so the last 3
tokens will be in this view:
```c++
            ggml_tensor * last_conv = ggml_view_3d(ctx0, conv_x,
                d_conv - 1,
                d_inner + 2*n_group*d_state,
                n_seqs,
                conv_x->nb[1],
                conv_x->nb[2],
                n_seq_tokens*(conv_x->nb[0]));
```
And this is then used to copy into the conv_states_all for the next timestep:
```c++
            ggml_build_forward_expand(gf,
                ggml_cpy(ctx0, last_conv,
                    ggml_view_1d(ctx0, conv_states_all,
                        (d_conv - 1)*(d_inner + 2*n_group*d_state)*(n_seqs),
                        kv_head*(d_conv - 1)*(d_inner + 2*n_group*d_state)*ggml_element_size(conv_states_all))));
```
So that is how the conv_states_all is updated for the next timestep.

After that we have the actual convolution operation:
```c++
            xBC = ggml_ssm_conv(ctx0, conv_x, model.layers[il].ssm_conv1d);
```
So the convolution kernel is:
```console
(gdb) p *model.layers[il].ssm_conv1d
$121 = {type = GGML_TYPE_F32, buffer = 0x5555579904f0, ne = {4, 3328, 1, 1}, nb = {4, 16, 53248, 53248}, op = GGML_OP_NONE,
  op_params = {0 <repeats 16 times>}, flags = 0, src = {0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0}, view_src = 0x0,
  view_offs = 0, data = 0x7ffed21b5060, name = "blk.0.ssm_conv1d.weight", '\000' <repeats 40 times>, extra = 0x0,
  padding = "\000\000\000\000\000\000\000"}
```
And what we are convolving over is:
```console
(gdb) p *conv_x
$122 = {type = GGML_TYPE_F32, buffer = 0x0, ne = {10, 3328, 1, 1}, nb = {4, 40, 133120, 133120}, op = GGML_OP_CONCAT, op_params = {
    0 <repeats 16 times>}, flags = 0, src = {0x55555628f2c0, 0x55555628fcd0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0}, 
  view_src = 0x0, view_offs = 0, data = 0x0, name = '\000' <repeats 63 times>, extra = 0x0, padding = "\000\000\000\000\000\000\000"}
```
So the dimensions are:
```console
conv_x: [10, 3328]  - 10 timesteps, 3328 features
kernel: [4, 3328]   - kernel size 4, one kernel per feature

output: [7, 3328]   - produces 7 outputs (one per new token)


Input (signal/sx) (conv_x):
[-3]  [-2]  [-1]    [0]  [1]  [2]  [3]  [4]  [5]  [6]
 from cache (3)          current tokens (7)

Kernel (ssm_conv1d): 
[k0, k1, k2, k3]


output[t] = k0 * conv_x[t-3] + k1 * conv_x[t-2] + k2 * conv_x[t-1] + k3 * conv_x[t]

[sx[-3], sx[-2], sx[-1], sx[0], sx[1], sx[2], sx[3], sx[4], sx[5], sx[6]]
 [k0     k1      k2      k3]

output[0] = k0 * conv_x[-3]  + k1 * conv_x[-2]  + k2 * conv_x[-1]  + k3 * conv_x[0]

[sx[-3], sx[-2], sx[-1], sx[0], sx[1], sx[2], sx[3], sx[4], sx[5], sx[6]]
          [k0     k1      k2     k3]
output[1] = k0 * conv_x[-2]  + k1 * conv_x[-1]  + k2 * conv_x[0]   + k3 * conv_x[1]

[sx[-3], sx[-2], sx[-1], sx[0], sx[1], sx[2], sx[3], sx[4], sx[5], sx[6]]
                  [k0     k1      k2     k3]
output[2] = k0 * conv_x[-1]  + k1 * conv_x[0]   + k2 * conv_x[1]   + k3 * conv_x[2]

[sx[-3], sx[-2], sx[-1], sx[0], sx[1], sx[2], sx[3], sx[4], sx[5], sx[6]]
                          [k0     k1      k2     k3]
output[3] = k0 * conv_x[0]   + k1 * conv_x[1]   + k2 * conv_x[2]   + k3 * conv_x[3]

[sx[-3], sx[-2], sx[-1], sx[0], sx[1], sx[2], sx[3], sx[4], sx[5], sx[6]]
                                [k0     k1      k2     k3]
output[4] = k0 * conv_x[1]   + k1 * conv_x[2]   + k2 * conv_x[3]   + k3 * conv_x[4]

[sx[-3], sx[-2], sx[-1], sx[0], sx[1], sx[2], sx[3], sx[4], sx[5], sx[6]]
                                       [k0     k1      k2     k3]
output[5] = k0 * conv_x[2]   + k1 * conv_x[3]   + k2 * conv_x[4]   + k3 * conv_x[5]


[sx[-3], sx[-2], sx[-1], sx[0], sx[1], sx[2], sx[3], sx[4], sx[5], sx[6]]
                                              [k0     k1      k2     k3]
output[6] = k0 * conv_x[3]   + k1 * conv_x[4]   + k2 * conv_x[5]   + k3 * conv_x[6]
```
Notice that for each step we take we are mixing in the previous three timesteps!
This is how local temporal context is added before the SSM computation.

This operation is performed for each of the 3328 dimensions (rows) in parallel.


_wip_
