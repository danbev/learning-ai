### DFlash2
Is an extension to dflash, similar to how dspark is an extension, and addresses
the multi-modal collisions and suffix acceptance decay in block diffusion.

There is an issue with dflash which is inherent to the parallel processing which
is, when we do parallel decoding we are predicting the complete "sentence" in
one go. There is no way for any token to know about the token prior to it or
after it as they get generated at the same time. And just like dspark is an
extension that addresses this issue, dflash2 also addresses this issue but in a
different way.

Standard non-causal self-attention lacks a strong local directional constraint,
causing the draft quality to degrade toward the end of the block (suffix decay).

So dspark solved this with a small network layer where tokens are are
sequentially passed through, enabling them to take the token before them into
account. This does have a cost and this is what dflash2 tries to avoid. Instead
of just drafting a single token for each posistion it will draft k tokens.
This is like a matrix where there is a row of k tokens for each position. And
then we can think of this as each one of these having connections to each other,
and there are probabilities for each paths (this would normally be static from
perhaps a vocabulary count or something but it is dynamic in this case). And these
are used to "walk" throw this matrix (is makes what makes it a lattice) and we
will get a score for each path. The path with the highest score contains the
tokens that are the most probably.

So instead of just drafting a single token for each position, DFlash2 drafts k
top candidate tokens for every slot in parallel, forming an n x k grid
(a "lattice" or trellis).
So instead of:
```console
pos0 : token
pos1 : token
pos2 : token

```

So in a single draft the model will output candidates for every position at the
same time, perhaps something like this:
```console
                  sequence pos 0        sequence pos 1        sequence pos 2
                     (i = 0)               (i = 1)               (i = 2)
                 +-------------+       +-------------+       +-------------+
candidate k=0    | [0] "new"   |       | [0] "red"   |       | [0] "car"   |
                 +-------------+       +-------------+       +-------------+

                 +-------------+       +-------------+       +-------------+
candidate k=1    | [1] "used"  |       | [1] "sports"|       | [1] "truck" |
                 +-------------+       +-------------+       +-------------+

                 +-------------+       +-------------+       +-------------+
candidate k=2    | [2] "fresh" |       | [2] "fast"  |       | [2] "bike"  |
                 +-------------+       +-------------+       +-------------+
```
Notice that we have the token positions as the columns and the rows are the
top k candidates for each position. For example, pos 0 has the top 3 candidates:
"new", "used", "fresh".

```console
Position 0 Candidates                         Position 1 Candidates

┌─────────────────┐                           ┌─────────────────┐
│   [0] "new"     │─────────────┬────────────>│   [0] "red"     │
└─────────────────┘             │             └─────────────────┘
                                │
┌─────────────────┐             ├────────────>┌─────────────────┐
│   [1] "used"    │─────────────┼────────────>│   [1] "sports"  │
└─────────────────┘             │             └─────────────────┘
                                │
┌─────────────────┐             └────────────>┌─────────────────┐
│   [2] "fresh"   │──────────────────────────>│   [2] "fast"    │
└─────────────────┘                           └─────────────────┘
```
So we could pick any of pos 0 candidates, and each could pick any of the candidates
in the next position, giving as a 3 * 3 = 9 unique connections.

Now, multi-modal collisions happen because position i doesn't know which token
position i-1 chose. This creates "ambiguity", there are multiple possible token
combinations, but only some are grammatically coherent.
Unlike DSpark, which runs a small neural network sequentially token-by-token,
DFlash2 avoids running any sequential neural network layers on the GPU.

To resolve token ambiguity without running a sequential model pass:
1. A lightweight candidate selector evaluates a k x k transition matrix between
   adjacent positions (pos_i-1 -> pos_i).

```console
              pos1[0]"red"  pos1[1]"sports" pos1[2]"fast"
pos0[0]"new"   [  1.2            3.5              0.9  ]
pos0[1]"used"  [  0.8            2.0              0.4  ]
pos0[2]"fresh" [  0.5            0.1              1.1  ]
```

2. The transition scores are generated dynamically on-the-fly using the target model's
   projected hidden state h, steering the choices toward coherent phrasing.

3. A C++ runtime (dflash.cpp) walks the lattice using dynamic programming
   (Viterbi search) to find the path that maximizes total probability.

Result: Multi-modal collisions are eliminated, order is preserved, and suffix
acceptance rates remain high—all while keeping the draft phase 100% parallel.

### Two-Tap Dynamic Convolution
Tap refers to a filter coefficient, the kernel size. Dynamic means that the weights
are generated dynamically on the fly based on the current hidden state (not static).
So two-tap means k=2, that we have a convolution size of 2. So at any position
i in the draft the layer operates on the hidden representation of the current
token i and i-1.

### Suffix decay
This refers to a sharp drop in token acceptance rates as we move from the
beginning of the draft block, the prefix, to the end of the block, the suffix).
```
[anchor token,   d_0,  d_1, ..., d_n  ]
  pos0           pos1  pos2 ..., pos_n
```
The first position that comes after the anchor token, which is the token that
the target model actually predicted is usually very accurate, often an 85-90%
acceptance rate. Next, we have pos_2 which is trying to predict a token
conditioned on pos1 which has not been verified by the target model yet.
pos_n tries to predict a token based on n-1 unverified, hypothetical tokens.
Because every predicted slot carries a small margin of error, that uncertainty
compounds exponentially as you go deeper into the block. By Position 6, the
draft model is effectively trying to predict the future based on a stack of
guesses.

Standard autoregressive LLMs enforce strict causal attention: token i can only
look backward at token i-1, enforcing a strict left-to-right cause-and-effect chain.

In parallel block diffusion, the draft model uses non-causal (bidirectional)
attention so that all n tokens in the block can look at each other simultaneously
in a single forward pass.

In non-causal attention, Slot 4 attends to Slot 1, 2, 3, 5, and 6 all at once.
It treats the whole block as a pool of information rather than a strict left to
right timeline. Non-causal attention lacks a built-in mechanism to force Slot 4
to obey the exact output of Slot 3. Instead of predicting "What word specifically
follows Slot 3?", Slot 4 predicts "What word fits generally into this entire
6-token region?"

The draft model outputs tokens that look locally plausible in isolation, but
fail to form a strict, causal chain.


### dflash2 implementation
```c++
static void build_dflash2_selector(llm_graph_context & g, const llama_model & model, ggml_tensor * tokens) {
    ...

    ggml_tensor * candidates  = ggml_top_k(ctx0, res->t_logits, top_k);
    ggml_tensor * logits_rows = ggml_reshape_3d(ctx0, res->t_logits, 1, res->t_logits->ne[0], n_tokens);
    ggml_tensor * unary       = ggml_reshape_2d(ctx0, ggml_get_rows(ctx0, logits_rows, candidates),
                                                top_k, n_tokens);
```
Unary here a score that depends on only one thing in isolation. How good does
the model think candicate X is at this position, on its own.
So we call ggml_top_k which will create a tensor operation that will return the
indices of the top k candidates when the graph is later executed. And we then
use those indices to plick out the actual candidate values from the logits and
they will then be stored in the unary tensor.


If we look at draft in speculative.cpp, even if this might feel like the reverse
order this is actually what happens at runtime:
```c++
    void draft(common_speculative_draft_params_vec & dparams) override {
        auto & ctx_dft = params.ctx_dft;

        batch.clear();

        // build one batch holding every drafting sequence's noise block into a single decode)
        // record where each block starts and its size
        std::vector<int32_t> i_block_beg(n_seq, -1);
        std::vector<int32_t> n_block    (n_seq,  0);
```
```console
(gdb) p i_block_beg
$4 = std::vector of length 4, capacity 4 = {-1, -1, -1, -1}

(gdb) p n_block
$5 = std::vector of length 4, capacity 4 = {0, 0, 0, 0}

(gdb) p n_seq
$6 = 4
```
I'm runnin llama-server which has 4 sequences/slots.
Next we will iterate over all the sequences:
```c++
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            common_sampler_reset(smpls[seq_id].get());

            const int32_t n = (int32_t) dp.pos0;

            const int32_t n_draft = params.n_max;

            const int32_t n_block_tokens = n_draft + (is_dspark && sample_from_anchor ? 0 : 1);

            // store the start of this draft block in the batch. This will increment
            // for each token that we add below.
            i_block_beg[seq_id] = batch.size();
            // currently n_block_tokens is 8
            n_block    [seq_id] = n_block_tokens;

            // for each 8 token do the following.
            for (int32_t i = 0; i < n_block_tokens; ++i) {
                // use the current drafting params last token id if this is
                // first token as that is our anchor token and otherwise we add
                // the mask token id.
                batch.add(i == 0 ? dp.id_last : mask_token_id, n + i, seq_id, !is_dflash2);
                                                                 ↑               ↑
                                                                 pos            output
            }
        }
```
dparams are the draft params (one for each sequence):
```console
(gdb) ptype dparams[0]
type = struct common_speculative_draft_params {
    bool drafting;
    int32_t n_max;
    llama_pos pos0;
    llama_token id_last;
    const llama_tokens *prompt;
    llama_tokens *result;
    std::vector<std::vector<llama_token_data>> *result_q;
    float temp;
    uint32_t seed;
}

(gdb) p seq_id
$10 = 3

1262	            const int32_t n_draft = params.n_max;
(gdb) p n
$11 = 377

(gdb) p n_draft
$12 = 7

(gdb) p n_block_tokens
$13 = 8

(gdb) p i_block_beg
$16 = std::vector of length 4, capacity 4 = {-1, -1, -1, 0}

(gdb) p n_block[3]
$18 = 8

(gdb) p !is_dflash2
$23 = false
```
Notice that output is false for dflash2 but not for other version of dflash. The
others need to create output for each token in the batch. It only uses the
separate embd_nextn mechanicm and does not sampler and can avoid copying the
outputs for each token.

We then process the batch:
```c++
        // decode all sequence's noise block in a single batch
        int ret = llama_process(ctx_dft, LLAMA_PROCESS_TYPE_DECODE, batch.get());
```

After this we will again iterate over all the sequences but we will skip "inactive"
blocks which are the i_block_beg that have -1 as their index, so they were
nere set in the first iteration above:
```c++
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (i_block_beg[seq_id] < 0) {
                continue;
            }
            auto & dp = dparams[seq_id];

            const int32_t beg            = i_block_beg[seq_id];
            const int32_t n_block_tokens = n_block[seq_id];

            auto & result = *dp.result;

            if (is_dflash2) {
                const float * lattice = llama_get_embeddings_nextn(ctx_dft);
                GGML_ASSERT(lattice && "DFlash2 selector produced no lattice");

                int32_t predecessor = 0;

                // notice that we start iterating at 1, so we skip the first
                // token which is the anchor token.
                for (int32_t i = 1; i < n_block_tokens; ++i) {

                    const float * row = lattice + (size_t) (beg + i) * n_embd_dec;
```
TODO: dig into how this lattice is created. It is just a flat float pointer here
but the buffer it points to has a specific shape just the same.

Indexing into the lattice flat buffer but the shape it had was
[n_embd_dec, n_tokens] which would be [5120, 8] in our case. We can verify this
by printing the shape back in build_dflash2_selector:
```console
(gdb) p packed->ne
$31 = {5120, 8, 1, 1}
0  [0  ...     5119]
1  [0  ...     5119]
2  [0  ...     5119]
 ...
7  [0          5119]
```
So row is a point to the start of this memory of floats and we have rows of
5120 and 8 tokens in total. If we inspect one row we have the following layout:
```console
Flattened:
[0 ... 15 16 ...    271  ...  5119][ ...        ][...]
[tok ids][16x16 scores][0 padding ]↑
          
        row 0                          row 1

row 0:
[0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15]
     candidate token ids

  0  [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15]
  1  [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15]
           .
           .
           .
  15 [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15]

       transition score matrix
```
Lets inspect some values to verify this:
```console
(gdb) p row[0]@15
$108 = {3274, 1822, 3296, 1637, 1428, 279, 3134, 1156, 4145, 3377, 2693, 4087, 709, 17313, 43070}

gdb) p params.ctx_dft->model->vocab->pimpl->id_to_token[row[0]]
$113 = {text = "Ġtask", score = 0, attr = LLAMA_TOKEN_ATTR_NORMAL}
(gdb) p params.ctx_dft->model->vocab->pimpl->id_to_token[row[1]]
$114 = {text = "Ġmain", score = 0, attr = LLAMA_TOKEN_ATTR_NORMAL}
(gdb) p params.ctx_dft->model->vocab->pimpl->id_to_token[row[2]]
$115 = {text = "Ġquestion", score = 0, attr = LLAMA_TOKEN_ATTR_NORMAL}
(gdb) p params.ctx_dft->model->vocab->pimpl->id_to_token[row[3]]
$116 = {text = "Ġperson", score = 0, attr = LLAMA_TOKEN_ATTR_NORMAL}
(gdb) p params.ctx_dft->model->vocab->pimpl->id_to_token[row[4]]
$117 = {text = "Ġcurrent", score = 0, attr = LLAMA_TOKEN_ATTR_NORMAL}
(gdb) p params.ctx_dft->model->vocab->pimpl->id_to_token[row[5]]
$118 = {text = "Ġthe", score = 0, attr = LLAMA_TOKEN_ATTR_NORMAL}
(gdb) p params.ctx_dft->model->vocab->pimpl->id_to_token[row[6]]
$119 = {text = "Ġquery", score = 0, attr = LLAMA_TOKEN_ATTR_NORMAL}
(gdb) p params.ctx_dft->model->vocab->pimpl->id_to_token[row[7]]
$120 = {text = "Ġuser", score = 0, attr = LLAMA_TOKEN_ATTR_NORMAL}
(gdb) p params.ctx_dft->model->vocab->pimpl->id_to_token[row[8]]
$121 = {text = "Ġsimple", score = 0, attr = LLAMA_TOKEN_ATTR_NORMAL}
(gdb) p params.ctx_dft->model->vocab->pimpl->id_to_token[row[9]]
$122 = {text = "Ġproblem", score = 0, attr = LLAMA_TOKEN_ATTR_NORMAL}
(gdb) p params.ctx_dft->model->vocab->pimpl->id_to_token[row[10]]
$123 = {text = "Ġplayer", score = 0, attr = LLAMA_TOKEN_ATTR_NORMAL}
(gdb) p params.ctx_dft->model->vocab->pimpl->id_to_token[row[11]]
$124 = {text = "Ġanswer", score = 0, attr = LLAMA_TOKEN_ATTR_NORMAL}
(gdb) p params.ctx_dft->model->vocab->pimpl->id_to_token[row[12]]
$125 = {text = "Ġfunction", score = 0, attr = LLAMA_TOKEN_ATTR_NORMAL}
(gdb) p params.ctx_dft->model->vocab->pimpl->id_to_token[row[13]]
$126 = {text = "Ġassistant", score = 0, attr = LLAMA_TOKEN_ATTR_NORMAL}
(gdb) p params.ctx_dft->model->vocab->pimpl->id_to_token[row[14]]
$127 = {text = "Ġsimplest", score = 0, attr = LLAMA_TOKEN_ATTR_NORMAL}
(gdb) p params.ctx_dft->model->vocab->pimpl->id_to_token[row[15]]
$128 = {text = "Ġcaller", score = 0, attr = LLAMA_TOKEN_ATTR_NORMAL}

(gdb) p row[16]@256
$134 = {
  13.9317217, 14.9860287, 23.3110523, 15.2346439, 15.9410925, 13.727519, 17.1482162, 28.5675392, 15.9120941,
  13.466753, 13.5670815, 16.9694653, 13.6387215, 13.913413, 14.2112055, 13.6912098, 13.9317217, 14.9860287,
  23.3110523, 15.2346439, 15.9410925, 13.727519, 17.1482162, 28.5675392, 15.9120941, 13.466753, 13.5670815,
  16.9694653, 13.6387215, 13.913413, 14.2112055, 13.6912098, 13.9317217, 14.9860287, 23.3110523, 15.2346439,
  15.9410925, 13.727519, 17.1482162, 28.5675392, 15.9120941, 13.466753, 13.5670815, 16.9694653, 13.6387215,
  13.913413, 14.2112055, 13.6912098, 13.9317217, 14.9860287, 23.3110523, 15.2346439, 15.9410925, 13.727519,
  17.1482162, 28.5675392, 15.9120941, 13.466753, 13.5670815, 16.9694653, 13.6387215, 13.913413, 14.2112055,
  13.6912098, 13.9317217, 14.9860287, 23.3110523, 15.2346439, 15.9410925, 13.727519, 17.1482162, 28.5675392,
  15.9120941, 13.466753, 13.5670815, 16.9694653, 13.6387215, 13.913413, 14.2112055, 13.6912098, 13.9317217,
  14.9860287, 23.3110523, 15.2346439, 15.9410925, 13.727519, 17.1482162, 28.5675392, 15.9120941, 13.466753,
  13.5670815, 16.9694653, 13.6387215, 13.913413, 14.2112055, 13.6912098, 13.9317217, 14.9860287, 23.3110523,
  15.2346439, 15.9410925, 13.727519, 17.1482162, 28.5675392, 15.9120941, 13.466753, 13.5670815, 16.9694653,
  13.6387215, 13.913413, 14.2112055, 13.6912098, 13.9317217, 14.9860287, 23.3110523, 15.2346439, 15.9410925,
  13.727519, 17.1482162, 28.5675392, 15.9120941, 13.466753, 13.5670815, 16.9694653, 13.6387215, 13.913413,
  14.2112055, 13.6912098, 13.9317217, 14.9860287, 23.3110523, 15.2346439, 15.9410925, 13.727519, 17.1482162,
  28.5675392, 15.9120941, 13.466753, 13.5670815, 16.9694653, 13.6387215, 13.913413, 14.2112055, 13.6912098,
  13.9317217, 14.9860287, 23.3110523, 15.2346439, 15.9410925, 13.727519, 17.1482162, 28.5675392, 15.9120941,
  13.466753, 13.5670815, 16.9694653, 13.6387215, 13.913413, 14.2112055, 13.6912098, 13.9317217, 14.9860287,
  23.3110523, 15.2346439, 15.9410925, 13.727519, 17.1482162, 28.5675392, 15.9120941, 13.466753, 13.5670815,
  16.9694653, 13.6387215, 13.913413, 14.2112055, 13.6912098, 13.9317217, 14.9860287, 23.3110523, 15.2346439,
  15.9410925, 13.727519, 17.1482162, 28.5675392, 15.9120941, 13.466753, 13.5670815, 16.9694653, 13.6387215,
  13.913413, 14.2112055, 13.6912098, 13.9317217, 14.9860287, 23.3110523, 15.2346439, 15.9410925, 13.727519,
  17.1482162, 28.5675392, 15.9120941, 13.466753, 13.5670815, 16.9694653, 13.6387215, 13.913413, 14.2112055,
  13.6912098, 13.9317217, 14.9860287, 23.3110523, 15.2346439, 15.9410925, 13.727519, 17.1482162, 28.5675392,
  15.9120941, 13.466753, 13.5670815, 16.9694653, 13.6387215, 13.913413, 14.2112055, 13.6912098, 13.9317217,
  14.9860287, 23.3110523, 15.2346439, 15.9410925, 13.727519, 17.1482162, 28.5675392, 15.9120941, 13.466753,
  13.5670815, 16.9694653, 13.6387215, 13.913413, 14.2112055, 13.6912098, 13.9317217, 14.9860287, 23.3110523,
  15.2346439, 15.9410925, 13.727519, 17.1482162, 28.5675392, 15.9120941, 13.466753, 13.5670815, 16.9694653,
  13.6387215, 13.913413, 14.2112055, 13.6912098}

(gdb) p row[16+256]@10
$112 = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
```
If we round the 16x16 to decimals and arange them in a 16x16 grid we get:
```console
[13.93, 14.99, 23.31, 15.23, 15.94, 13.73, 17.15, 28.57, 15.91, 13.47, 13.57, 16.97, 13.64, 13.91, 14.21, 13.69]
[13.93, 14.99, 23.31, 15.23, 15.94, 13.73, 17.15, 28.57, 15.91, 13.47, 13.57, 16.97, 13.64, 13.91, 14.21, 13.69]
[13.93, 14.99, 23.31, 15.23, 15.94, 13.73, 17.15, 28.57, 15.91, 13.47, 13.57, 16.97, 13.64, 13.91, 14.21, 13.69]
[13.93, 14.99, 23.31, 15.23, 15.94, 13.73, 17.15, 28.57, 15.91, 13.47, 13.57, 16.97, 13.64, 13.91, 14.21, 13.69]
[13.93, 14.99, 23.31, 15.23, 15.94, 13.73, 17.15, 28.57, 15.91, 13.47, 13.57, 16.97, 13.64, 13.91, 14.21, 13.69]
[13.93, 14.99, 23.31, 15.23, 15.94, 13.73, 17.15, 28.57, 15.91, 13.47, 13.57, 16.97, 13.64, 13.91, 14.21, 13.69]
[13.93, 14.99, 23.31, 15.23, 15.94, 13.73, 17.15, 28.57, 15.91, 13.47, 13.57, 16.97, 13.64, 13.91, 14.21, 13.69]
[13.93, 14.99, 23.31, 15.23, 15.94, 13.73, 17.15, 28.57, 15.91, 13.47, 13.57, 16.97, 13.64, 13.91, 14.21, 13.69]
[13.93, 14.99, 23.31, 15.23, 15.94, 13.73, 17.15, 28.57, 15.91, 13.47, 13.57, 16.97, 13.64, 13.91, 14.21, 13.69]
[13.93, 14.99, 23.31, 15.23, 15.94, 13.73, 17.15, 28.57, 15.91, 13.47, 13.57, 16.97, 13.64, 13.91, 14.21, 13.69]
[13.93, 14.99, 23.31, 15.23, 15.94, 13.73, 17.15, 28.57, 15.91, 13.47, 13.57, 16.97, 13.64, 13.91, 14.21, 13.69]
[13.93, 14.99, 23.31, 15.23, 15.94, 13.73, 17.15, 28.57, 15.91, 13.47, 13.57, 16.97, 13.64, 13.91, 14.21, 13.69]
[13.93, 14.99, 23.31, 15.23, 15.94, 13.73, 17.15, 28.57, 15.91, 13.47, 13.57, 16.97, 13.64, 13.91, 14.21, 13.69]
[13.93, 14.99, 23.31, 15.23, 15.94, 13.73, 17.15, 28.57, 15.91, 13.47, 13.57, 16.97, 13.64, 13.91, 14.21, 13.69]
[13.93, 14.99, 23.31, 15.23, 15.94, 13.73, 17.15, 28.57, 15.91, 13.47, 13.57, 16.97, 13.64, 13.91, 14.21, 13.69]
[13.93, 14.99, 23.31, 15.23, 15.94, 13.73, 17.15, 28.57, 15.91, 13.47, 13.57, 16.97, 13.64, 13.91, 14.21, 13.69]
```
Each values in this matrix, [row][col] is a score of
```console
Score(u -> v) = unary(v) + e_pred(u)^T W_h h_t . e_succ(v)

Candidate u at step t-1 is immediately followed by candidate v at step t.
```
This score evalutate how valid/likely/coherent it is for token v to follow token
u, taking into account both the local prediction and the transistion betwen them.
And recall that what we are trying to address is that with just the pure diffusion
the probabilites doen't have any knowledge/influence about each other so if we
were to just pick the hightest probabilites for each token position it might not
produce a coherent sequence and would most likley be rejected by the token
verification process. So 

So we have two distinct parts:
```console
Score(u -> v) = unary(v) + pairwise(u, v)
```

v is how plausable is candidate v at position t based soley on the prompt and
the drafting features. This is just the standard logit for candidate v.
The pairwise is given that we picked u (prev) right before this, does v (next)
make sense as the next token?

So DFlash2 has top-16. We first have the single forward pass where the
diffusion drafter predicts distributions for all T positions at the same time.
And like we discussed taking the top-1 at every position often fails because
adjcent positions don't coordinate which leads to grammar clashes and fails
target model verification. What was found in practice (emperical finding) was
that even if the top-1 guess is wrong the correct token is almost always in the
top 16. So that is why DFlash2 uses top-16.
So we have the following:
```console
t-1   : top 16 candidates {u₁, u₂, ..., u₁₆}
t     : top 16 candidates {v₁, v₂, ..., v₁₆}
```
And in our current case we have 8 tokens, so each of them will get a 16x16
matrix which contains the information for the current token and the current and
the score of transitioning from the previous token to the current token. And this
is part of the drafters computation graph, in the build_dflash2_selector where
I skipped this before.

So lets take an example, what does lattice[0, 2] mean?  
The row index 0 is the candidate choice number 0 at position t-1. And the column
index 2 is the candidate choice number 2 at postion t.
```console
[13.93, 14.99, 23.31, 15.23, 15.94, 13.73, 17.15, 28.57, 15.91, 13.47, 13.57, 16.97, 13.64, 13.91, 14.21, 13.69]
                 ↑
                 2
```
The score here is the score of transitioning from candidate number 0 to candidate
2. So the score of going from the first pick of the previous token to the second
pick of the current token.

So with that background information lets take a look at the loop here, we have
the row which we have already discusses and the scores pointer needs to skip
the 16 initial token ids that each row has.
```c++
                int32_t predecessor = 0;
                for (int32_t i = 1; i < n_block_tokens; ++i) {
                    const float * row = lattice + (size_t) (beg + i) * n_embd_dec;
                    const float * scores = row + selector_top_k + (size_t) predecessor * selector_top_k;

                    predecessor = (int32_t) std::distance(scores,
                            std::max_element(scores, scores + selector_top_k));

                    if (params.p_min > 0.0f) {
                        // softmax(scores) at the argmax, i.e. 1 / sum(exp(s_k - s_max))
                        float sum = 0.0f;
                        for (int32_t k = 0; k < selector_top_k; ++k) {
                            sum += std::exp(scores[k] - scores[predecessor]);
                        }
                        if (1.0f / sum < params.p_min) {
                            break;
                        }
                    }
                    result.push_back((llama_token) row[predecessor]);
                }

                if (result.size() < (size_t) params.n_min) {
                    result.clear();
                }
                continue;
            }

            auto * smpl = smpls[seq_id].get();
```
So this is first calculating the number of hops from the first score, the 
start of the scores, to the maximum element in scores from the start, and the
end is specified as scores + 16 so that would be one row.

Notice that predecessor is initially zero, but for the next iterations it will
be updated to contain the max arg value's index. So later iterations will be
using the choses index of the previous iteration. This is how the "path" or
greedy walk is enabled.

Where max_elements will return an iterator to the largest element:
```console
(gdb) p scores[0]@16
$142 = {13.9317217, 14.9860287, 23.3110523, 15.2346439, 15.9410925, 13.727519, 17.1482162, 28.5675392, 15.9120941, 
  13.466753, 13.5670815, 16.9694653, 13.6387215, 13.913413, 14.2112055, 13.6912098}

(gdb) p std::max_element<float const*>((const float*)scores, (const float*)scores + selector_top_k)
$140 = (const float *) 0xfff2441cf05c

(gdb) p *(std::max_element<float const*>((const float*)scores, (const float*)scores + selector_top_k))
$141 = 28.5675392

(gdb) p $140 - (const float*) scores
$143 = 7
```
And we then uses std::distance with the score to get the index of this.
```console
(gdb) p predecessor
$144 = 7
```
Then we have a check if params.p_min is greater that 0.0f which it is not in
our case. And this is specified using `--spec-draft-p-min` and the drafters
prob for its top candidate falls below p_min it stops adding and breaks out
of the loop.

In this case we skip that and just add the index of the max arg value
, which just get from the first 0-15 values in the row:
```c++
                    result.push_back((llama_token) row[predecessor]);
```

result:
```console
(gdb) p result
$147 = std::vector of length 1, capacity 1 = {1156}
```
We then continue and to the same for all of the remaining 7 blocks.

