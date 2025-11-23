### output_ids
This is a field/member of llama_context:
```c++
struct llama_context {
    ...
    std::vector<int32_t> output_ids; // map batch token positions to ids of the logits and embd buffers
```
The index into output_ids corresponds to a token's index in the original
llama_batch that the user provided.

When the contructor runs it will call `reserve_outputs`:
```c++
            // resized during inference when a batch uses more outputs
            if (output_reserve(params.n_seq_max) < params.n_seq_max) {
                throw std::runtime_error("failed to reserve initial output buffer");
            }
```
And in output_reserve we have:
```c++
    if (output_ids.empty()) {
        // init, never resized afterwards
        output_ids.resize(n_batch);
    }
```
So this will resize the output_ids and initialize them to 0:
```console
(gdb) p n_batch
$4 = 2048
(gdb) p output_ids.size()
$5 = 2048
(gdb) p output_ids[0]
$7 = 0
(gdb) p output_ids[1]
```

And a bit further down in this function we have:
```c++
    // set all ids as invalid (negative)
    std::fill(output_ids.begin(), output_ids.end(), -1);
```
Notice that this will set/reset all the values in output_ids to -1 and this
happens whenever output_reserve is called.

So this list is initialized to -1 for all tokens in the batch.

Now, to fully understand how this is poplated and use we have to start looking
at llama_decode and were initialize the batch allocator (balloc):
```c++
    const bool output_all = cparams.embeddings;

    if (!balloc->init(batch_inp, vocab, memory.get(), n_embd, cparams.kv_unified ? LLAMA_MAX_SEQ : cparams.n_seq_max, output_all)) {
        LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
        return -1;
    }
```
The examples that I'm stepping through is [simple-prompt-multi](../fundamentals/llama.cpp/srcsimple-prompt-multi.cpp)
and batch_inp has 10 tokens:
```console
(gdb) p batch_inp
$1 = (const llama_batch &) @0x7fffffffd220: {n_tokens = 10, token = 0x5555559e32b0, embd = 0x0, pos = 0x555555a26920, 
  n_seq_id = 0x555555a40690, seq_id = 0x55555636df70, logits = 0x5555559dbf70 ""}
```
Which is split into two sequences. The first sequence has three tokens:
```console
(gdb) p vocab->pimpl->id_to_token[batch_inp.token[0]]
$5 = {text = "<s>", score = 0, attr = LLAMA_TOKEN_ATTR_CONTROL}
(gdb) p vocab->pimpl->id_to_token[batch_inp.token[1]]
$20 = {text = "▁Hello", score = -14784, attr = LLAMA_TOKEN_ATTR_NORMAL}
(gdb) p vocab->pimpl->id_to_token[batch_inp.token[2]]
$19 = {text = "▁", score = -1e+09, attr = LLAMA_TOKEN_ATTR_NORMAL}
(gdb) p batch_inp.seq_id[0][0]
$21 = 0
(gdb) p batch_inp.seq_id[1][0]
$22 = 0
(gdb) p batch_inp.seq_id[2][0]
$23 = 0
```
And the second sequence has seven tokens:
```console
(gdb) p vocab->pimpl->id_to_token[batch_inp.token[3]]
$32 = {text = "<s>", score = 0, attr = LLAMA_TOKEN_ATTR_CONTROL}
(gdb) p vocab->pimpl->id_to_token[batch_inp.token[4]]
$33 = {text = "▁Dan", score = -3692, attr = LLAMA_TOKEN_ATTR_NORMAL}
(gdb) p vocab->pimpl->id_to_token[batch_inp.token[5]]
$34 = {text = "▁lov", score = -12096, attr = LLAMA_TOKEN_ATTR_NORMAL}
(gdb) p vocab->pimpl->id_to_token[batch_inp.token[6]]
$35 = {text = "es", score = -8, attr = LLAMA_TOKEN_ATTR_NORMAL}
(gdb) p vocab->pimpl->id_to_token[batch_inp.token[7]]
$36 = {text = "▁ice", score = -14631, attr = LLAMA_TOKEN_ATTR_NORMAL}
(gdb) p vocab->pimpl->id_to_token[batch_inp.token[8]]
$37 = {text = "▁cre", score = -648, attr = LLAMA_TOKEN_ATTR_NORMAL}
(gdb) p vocab->pimpl->id_to_token[batch_inp.token[9]]
$38 = {text = "am", score = -55, attr = LLAMA_TOKEN_ATTR_NORMAL}
(gdb) p vocab->pimpl->id_to_token[batch_inp.token[10]]

(gdb) p batch_inp.seq_id[3][0]
$24 = 1
(gdb) p batch_inp.seq_id[4][0]
$25 = 1
(gdb) p batch_inp.seq_id[5][0]
$26 = 1
(gdb) p batch_inp.seq_id[6][0]
$27 = 1
(gdb) p batch_inp.seq_id[7][0]
$28 = 1
(gdb) p batch_inp.seq_id[8][0]
$29 = 1
(gdb) p batch_inp.seq_id[9][0]
$30 = 1
```

```c++
bool llama_batch_allocr::init(
        const llama_batch & batch_inp,
        const llama_vocab & vocab,
        const llama_memory_i * memory,
        uint32_t n_embd,
        uint32_t n_seq_max,
        bool output_all) {
    clear();

    batch = batch_inp;
```
clear will reset the batch allocators state, and notice that it stores the
batch_input for each decode. This state if first cleared along with other
fields:
```c++
void llama_batch_allocr::clear() {
    n_outputs = 0;

    batch = {};

    pos       .clear();
    n_seq_id  .clear();
    seq_id    .clear();
    seq_id_unq.clear();
    output    .clear();

    for (auto & cur : seq_pos) {
        cur.clear();
    }

    for (auto & cur : seq_cpl) {
        std::fill(cur.begin(), cur.end(), false);
    }

    seq_set.clear();

    seq_set_map.clear();

    std::fill(seq_idx.begin(), seq_idx.end(), -1);
}
```
Focusing on output releated logic we have the following:
```c++
    if (!batch.logits) {
        if (output_all) {
            // return the output for all tokens
            output.resize(batch.n_tokens, true);
        } else {
            // return the output only for the last token
            output.resize(batch.n_tokens, false);
            output[output.size() - 1] = true;
        }

        batch.logits = output.data();
    } else if (output_all) {
```
If the input batch has not provided the logits field then this is populated
and the last token is marked as an output. This is not the case for this session.

Next the number of outputs is calculated, which is simply checking the logits
field of each token:
```c++
    // count the outputs in this batch
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        n_outputs += batch.logits[i] != 0;
    }
```
```console
(gdb) p n_outputs
$45 = 2
```
Most of the rest of this function is related to sequence handling and verification
so I'm skipping that for now. But I do want to point out that at the end of this
function:
```c++
    split_reset();

    return true;
}

void llama_batch_allocr::split_reset() {
    out_ids.clear();

    n_used = 0;

    used.clear();
    used.resize(get_n_tokens(), false);
}
```
While we have not seen them be used get the fields will become important later:
```console
(gdb) ptype this.out_ids
type = std::vector<int>

(gdb) ptype this.used
type = std::vector<bool>
```

So back in llama_decode:
```c++
    const uint32_t n_tokens_all  = balloc->get_n_tokens();
    const uint32_t n_outputs_all = balloc->get_n_outputs();

    ...
    output_swaps.clear();
    ...

    while (true) {
        mctx = memory->init_batch(*balloc, cparams.n_ubatch, output_all);
```
```c++
llama_memory_context_ptr llama_kv_cache::init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) {
    GGML_UNUSED(embd_all);

    do {
        balloc.split_reset();

        std::vector<llama_ubatch> ubatches;

        while (true) {
            auto ubatch = n_stream == 1 ? balloc.split_simple(n_ubatch) : balloc.split_equal(n_ubatch, true);

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }
```
We say the split_reset earlier which clears out the out_ids and used fields.
Now, in our case we have n_stream = 2:
```console
(gdb) p n_stream
$47 = 2
```
So this will be a split creating micro batched with an equal number of sequences
```c++
llama_ubatch llama_batch_allocr::split_equal(uint32_t n_ubatch, bool sequential) {
    ...

    // concat the per-sequence-set lists
    std::vector<int32_t> idxs;

    for (uint32_t s = 0; s < n_seqs; ++s) {
        idxs.insert(idxs.end(), idxs_per_seq[s].begin(), idxs_per_seq[s].end());
    }

    return ubatch_add(idxs, n_seqs, true);
}
```

```console
(gdb) p seq_set
$50 = std::vector of length 10, capacity 16 = {
std::bitset = {[0] = 1},
std::bitset = {[0] = 1},
std::bitset = {[0] = 1},
std::bitset = {[1] = 1},
std::bitset = {[1] = 1},
std::bitset = {[1] = 1},
std::bitset = {[1] = 1},
std::bitset = {[1] = 1},
std::bitset = {[1] = 1},
std::bitset = {[1] = 1}}

(gdb) p used
$48 = std::vector<bool> of length 10, capacity 64 = {false, false, false, false, false, false, false, false, false, false}

(gdb) p idxs
$59 = std::vector of length 6, capacity 6 = {0, 1, 2, 3, 4, 5}

(gdb) p n_seqs
$60 = 2
```
What is happening here is that it is figuring out how to take one token from
each sequence at a time.

```c++
llama_ubatch llama_batch_allocr::ubatch_add(const std::vector<int32_t> & idxs, uint32_t n_seqs, bool equal_seqs) {
    const uint32_t n_tokens = idxs.size();

    auto udata = std::make_shared<llama_ubatch::data_t>();

    udata->token     .resize(n_tokens);
    udata->embd      .resize(n_embd_all);
    udata->pos       .resize(n_pos_all);
    udata->n_seq_id  .resize(n_tokens);
    udata->seq_id    .resize(n_tokens);
    udata->seq_id_unq.resize(0);
    udata->seq_idx   .resize(LLAMA_MAX_SEQ, -1);
    udata->output    .resize(n_tokens);
    ...
```
```c++
    struct data_t {
        std::vector<llama_token>    token;
        std::vector<float>          embd;
        std::vector<llama_pos>      pos;
        std::vector<int32_t>        n_seq_id;
        std::vector<llama_seq_id *> seq_id;
        std::vector<llama_seq_id>   seq_id_unq;
        std::vector<int32_t>        seq_idx;
        std::vector<int8_t>         output;
    };
```
```c++
    for (size_t i = 0; i < idxs.size(); ++i) {
        if (batch.token) {
            udata->token[i] = batch.token[idxs[i]];
        }

        if (batch.embd) {
            memcpy(udata->embd.data() + i*n_embd, batch.embd + (int64_t) idxs[i]*n_embd, n_embd*sizeof(float));
        }

        for (size_t j = 0; j < (size_t)n_pos_per_embd; ++j) {
            // if we are using M-RoPE
            //     if the current batch is text, we need to broadcast the same position across all RoPE sections
            //     otherwise, the input batch is image embeddings, we copy the positions as-is
            // if we are not using M-RoPE, there is only one position per token (this loop runs only once)
            size_t src_off = batch.token ? 0 : j*batch.n_tokens;
            udata->pos[j*n_tokens + i] = batch.pos[src_off + idxs[i]];
        }

        udata->n_seq_id[i] = batch.n_seq_id[idxs[i]];
        udata->seq_id[i]   = batch.seq_id[idxs[i]];
        udata->output[i]   = batch.logits[idxs[i]];

        for (int s = 0; s < udata->n_seq_id[i]; ++s) {
            seq_set_unq.set(udata->seq_id[i][s]);
        }

        if (udata->output[i]) {
            out_ids.push_back(idxs[i]);
        }
    }
```
Notice that this is populating the ubatch data output and it checks if the
current token being iterated is marked as output, in which case it add this
the out_ids list:
```console
(gdb) p out_ids
$66 = std::vector of length 1, capacity 1 = {2}
```
In our case this is interesting as out first sequence only has 3 tokens and
the last one is indeed set as an output. But this will not be the case for the
second sequence which has 7 tokens and only 3 are included in this micro batch
since we are taking three tokens from each sequence.
The final result, the ubatch created is:
```console
(gdb) p res
$67 = {b_equal_seqs = 1, n_tokens = 6, n_seq_tokens = 3, n_seqs = 2, n_seqs_unq = 2, n_pos = 1, token = 0x555555aab7e0, embd = 0x0,
  pos = 0x555555aab860, n_seq_id = 0x555555aab880, seq_id = 0x5555559e0870, seq_id_unq = 0x555555aab900, seq_idx = 0x555556367e10,
  output = 0x555555aab8a0 "", data = std::shared_ptr<llama_ubatch::data_t> (use count 1, weak count 0) = {get() = 0x555555a415a0}}

(gdb) p out_ids
$68 = std::vector of length 1, capacity 1 = {2}
```
Back in llama_kv_cache::init_batch we add the micro batch to the list of
micro batches:
```c++
    ubatches.push_back(std::move(ubatch)); // NOLINT
```
And then we continue the loop.
This time idxs will be:
```console
(gdb) p idxs
$69 = std::vector of length 4, capacity 4 = {6, 7, 8, 9}
```
And for these tokens, which are the remaining tokens from the second sequence
only the last token is marked as output:
```console
(gdb) p out_ids
$73 = std::vector of length 2, capacity 2 = {2, 9}
```
So, batch out_ids contains the original indices of the tokens that are marked
as output. This is what a user would use with llama_get_logits_ith for example.

After all that we will return to llama_decode where we have a call to
output_reserve:
```c++
    // reserve output buffer
    if (output_reserve(n_outputs_all) < n_outputs_all) {
        LLAMA_LOG_ERROR("%s: could not reserve space for batch with %d outputs\n", __func__, n_outputs_all);
        return -2;
    };
```
And this is where a backend buffer is created for the output. This is important
as the memory needs to be pinned in order to do efficient async copies from the
GPU.

Then we have the main do/while loop which will process the input batch, but
splittig it into micro batches to be processed by the backend.
```c++
    do {
        const auto & ubatch = mctx->get_ubatch();

        // count the outputs in this ubatch
        {
            int32_t n_outputs_new = 0;

            // If all tokens have their output/logits field set to true then
            // we can just use n_tokens directly.
            if (n_outputs_all == n_tokens_all) {
                n_outputs_new = ubatch.n_tokens;
            } else {
                // otherwise we need to count them
                for (uint32_t i = 0; i < ubatch.n_tokens; i++) {
                    n_outputs_new += (int32_t) (ubatch.output[i] != 0);
                }
            }

            // needs to happen before the graph is built
            n_outputs = n_outputs_new;
        }
```
Then the ubatch is processed:
```c++
        ggml_status status;
        const auto * res = process_ubatch(ubatch, LLM_GRAPH_TYPE_DECODER, mctx.get(), status);
```
And recall for the first micro batch we only have one output which is for the
first sequence:
```console
(gdb) p n_outputs
$78 = 1
```
So we will create an async copy operation which is why the pinned memory is needed
as this memory area should not be changed (paged out) while the GPU before the
GPU has written the data to it:
```c++
        // extract logits
        if (t_logits && n_outputs > 0) {
            ggml_backend_t backend_res = ggml_backend_sched_get_tensor_backend(sched.get(), t_logits);
            GGML_ASSERT(backend_res != nullptr);
            GGML_ASSERT(logits != nullptr);

            float * logits_out = logits + n_outputs_prev*n_vocab;

            if (n_outputs) {
                GGML_ASSERT( n_outputs_prev + n_outputs <= n_outputs_all);
                GGML_ASSERT((n_outputs_prev + n_outputs)*n_vocab <= (int64_t) logits_size);
                ggml_backend_tensor_get_async(backend_res, t_logits, logits_out, 0, n_outputs*n_vocab*sizeof(float));
            }
        }
```
And notice that we are using n_outputs as an index into this pinned memory
buffer. So we are writing the output to n_outputs_prev, which is currently 0 but
we will see for the second micro batch it will be 1. This is updated at the
end of the loop:
```c++
        n_outputs_prev += n_outputs;
    } while (mctx->next());
```
So that was one but we have another micro batch to process which will also have
a single output. This time the index in the the pinned memory buffer will be 1:
```console
(gdb) p n_outputs_prev
$81 = 1
```
And this is multiplied by the vocab size to get the correct offset into the logits
buffer.

And that completes the do/while loop processing all the micro batches. So we
had two outputs.

After decoding we have the following:
```c++
    n_outputs = n_outputs_all;

    // set output mappings
    if (n_outputs > 0) {
        bool sorted_output = true;

        auto & out_ids = balloc->get_out_ids();

        GGML_ASSERT(out_ids.size() == (size_t) n_outputs);

        for (int64_t i = 0; i < n_outputs; ++i) {
            int64_t out_id = out_ids[i];
            output_ids[out_id] = i;
            if (out_id != i) {
                sorted_output = false;
            }
        }
```
And in this case out_ids is the same as what we pushed into it earlier:
```console
(gdb) p out_ids
$83 = std::vector of length 2, capacity 2 = {2, 9}
```
Then we iterate over all the outputs:
```c++
        for (int64_t i = 0; i < n_outputs; ++i) {
            int64_t out_id = out_ids[i];
            output_ids[out_id] = i;
            if (out_id != i) {
                sorted_output = false;
            }
        }
```
And recall that output_ids are all -1 at this stage.
```console
(gdb) p output_ids
$84 = std::vector of length 1024, capacity 1024 = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,

(gdb) p out_id
$85 = 2

(gdb) p output_ids
$87 = std::vector of length 1024, capacity 1024 = {-1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
```
Notice that this as updated output_ids[2] = 0. And the second iteration will update
entry 9 to be 1:
```console
(gdb) p output_ids
$88 = std::vector of length 1024, capacity 1024 = {-1, -1, 0, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1,
                                                                                      ↑
```
Now, a user can specify the output indexes that it originally set, that is 2
and 9 to get the logits for those tokens. And this can be used to index into
the outputs_ids vector and use that value to get the correct row in the logits
buffer.
```c++
llama_get_logits_ith(ctx, 2);
```
```c++
float * llama_get_logits_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_logits_ith(i);
}
```

```c++
float * llama_context::get_logits_ith(int32_t i) {
    int64_t j = -1;

    output_reorder();

    try {
        if (logits == nullptr) {
            throw std::runtime_error("no logits");
        }

        if (i < 0) {  // for negative indices
            j = n_outputs + i;
            if (j < 0) {
                throw std::runtime_error(format("negative index out of range [0, %d)", n_outputs));
            }
        } else if ((size_t) i >= output_ids.size()) {
            throw std::runtime_error(format("out of range [0, %zu)", output_ids.size()));
        } else {
            // this is what our case will be, and like we mentioned this will
            // set j, the index into the pinned memory logits buffer to the value
            // of the output_ids at index i (output_ids[2] = 0)
            j = output_ids[i];
        }

        if (j < 0) {
            throw std::runtime_error(format("batch.logits[%d] != true", i));
        }
        if (j >= n_outputs) {
            // This should not happen
            throw std::runtime_error(format("corrupt output buffer (j=%" PRId64 ", n_outputs=%d)", j, n_outputs));
        }

        // This is were the actual indexing into the pinned memory logits buffer happens
        return logits + j*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid logits id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}
```

Here, logits_out is pointing to the start of the logits buffer for this batch.
And logits is the start of our pinned memory buffer. So we will copy the tensor
data from the GPU that is in t_logits to the pinned memory location pointed to
by logits_out.

### reordering
Lets say we have a batch with three tokens marked as outputs, indices 2, 5, and 8.
Conceptually you want:
```
Row 0 → batch index 2
Row 1 → batch index 5
Row 2 → batch index 8
```
The ggml graph computation may return the outputs in some backend optimized order
for efficiency, lets say (8, 2, 5).
So when the decode loop finished we would have populated output_ids like this:
```
output_ids[8] = 0
output_ids[2] = 1
output_ids[5] = 2
```
So if we wanted to get the logits for batch index 5 we would look up output_ids[5]
and get 2 which tells us that the logits for that token are in row 2 of the
logits buffer.

### backend sampling
For backend sampling we also need to use pinned memory for the sampled tokens,
logits, probablities, and for the candidate tokens.

What happens in llama_context outputs_reserve is that a buffer is allocated on
the backend and the size is normally detemined by the logits/embeddings needed.
For backend sampling we don't need them but we instead need to allocate some more
space for the extra information (the sampled tokens, probabilities, and candidates).


```console
$ nsys profile --trace=cuda --cuda-memory-usage=trune -o llama_profile ./build-gpu-sampler/bin/llama-cli -m models/Qwen2.5-VL-3B-Instruct-Q8_0.gguf --no-warmup --prompt 'What is the capital of Sweden?' -n 20  -no-cnv --backend-sampling --verbose-prompt -ngl 50 --backend-dist
```

```console
$ nsys stats --report cuda_api_sum llama_profile_prev.nsys-rep
Generating SQLite file llama_profile_prev.sqlite from llama_profile_prev.nsys-rep
Processing [llama_profile_prev.sqlite] with [/opt/nvidia/nsight-systems/2024.5.1/host-linux-x64/reports/cuda_api_sum.py]...

 ** CUDA API Summary (cuda_api_sum):

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)     Med (ns)    Min (ns)    Max (ns)    StdDev (ns)               Name
 --------  ---------------  ---------  ------------  -----------  ---------  -----------  ------------  -----------------------------
     82.3    1,454,598,159        654   2,224,156.2      6,735.0      1,197  146,549,333   6,764,963.6  cudaMemcpyAsync
      8.4      149,022,826        739     201,654.7      8,644.0        300      820,756     298,439.1  cudaStreamSynchronize
      5.6       99,136,786          5  19,827,357.2     45,443.0      6,725   98,997,592  44,257,509.9  cudaMemGetInfo
      2.4       42,766,907     15,203       2,813.1      1,556.0        403    4,581,633      53,918.2  cudaLaunchKernel
      0.4        7,488,829          2   3,744,414.5  3,744,414.5  1,058,022    6,430,807   3,799,132.7  cudaFreeHost
      0.3        5,124,321          2   2,562,160.5  2,562,160.5    916,016    4,208,305   2,327,999.9  cudaMallocHost
      0.2        3,055,249          4     763,812.3    153,442.0     10,930    2,737,435   1,318,072.4  cudaFree
      0.1        2,194,962          1   2,194,962.0  2,194,962.0  2,194,962    2,194,962           0.0  cudaGraphInstantiate_v12000
      0.1        1,038,266         23      45,142.0     12,547.0      8,741      322,905      79,720.4  cudaMalloc
      0.0          721,877          4     180,469.3    181,021.0    148,371      211,464      32,303.1  cudaGraphDestroy_v10000
      0.0          465,449          1     465,449.0    465,449.0    465,449      465,449           0.0  cudaGraphExecDestroy_v10000
      0.0          420,591          4     105,147.8     84,607.5     71,408      179,968      51,065.8  cudaGraphLaunch_v10000
      0.0          419,679          4     104,919.8     97,167.5     85,547      139,797      23,987.4  cudaGraphExecUpdate_v10020
      0.0          282,971          4      70,742.8     61,795.5     53,093      106,287      24,060.2  cudaStreamEndCapture_v10000
      0.0          253,929        121       2,098.6      1,655.0        272       25,986       2,502.0  cudaMemsetAsync
      0.0          142,183          2      71,091.5     71,091.5     61,194       80,989      13,997.2  cuMemSetAccess
      0.0          133,615          1     133,615.0    133,615.0    133,615      133,615           0.0  cuMemUnmap
      0.0          117,777          2      58,888.5     58,888.5     21,855       95,922      52,373.3  cuMemCreate
      0.0           76,740         36       2,131.7      1,192.5        988       19,955       3,304.9  cudaMemset
      0.0           42,951          4      10,737.8      7,659.5      4,342       23,290       8,556.9  cudaStreamBeginCapture_v10000
      0.0           19,284          1      19,284.0     19,284.0     19,284       19,284           0.0  cudaStreamCreateWithFlags
      0.0           11,706          1      11,706.0     11,706.0     11,706       11,706           0.0  cuMemAddressReserve
      0.0            6,777          2       3,388.5      3,388.5      3,023        3,754         516.9  cuMemMap
      0.0            6,477          1       6,477.0      6,477.0      6,477        6,477           0.0  cudaStreamDestroy
      0.0            5,823          1       5,823.0      5,823.0      5,823        5,823           0.0  cuMemAddressFree
      0.0            2,132          1       2,132.0      2,132.0      2,132        2,132           0.0  cuMemGetAllocationGranularity
      0.0              615          2         307.5        307.5        176          439         186.0  cuMemRelease
      0.0              516          1         516.0        516.0        516          516           0.0  cuModuleGetLoadingMode

```

```console
rocessing [/tmp/nsys-report-dc9b.sqlite] with [/opt/nvidia/nsight-systems/2024.5.1/host-linux-x64/reports/cuda_api_sum.py]...

 ** CUDA API Summary (cuda_api_sum):

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)     Med (ns)    Min (ns)    Max (ns)    StdDev (ns)               Name
 --------  ---------------  ---------  ------------  -----------  ---------  -----------  ------------  -----------------------------
     75.8    1,337,821,503        654   2,045,598.6      5,417.0      1,161  146,842,944   6,748,645.3  cudaMemcpyAsync
     15.1      267,468,325        739     361,932.8      8,814.0        335    7,752,546   1,008,380.2  cudaStreamSynchronize
      5.6       98,436,860          5  19,687,372.0     45,473.0      6,819   98,278,222  43,933,625.7  cudaMemGetInfo
      2.4       42,801,321     15,203       2,815.3      1,666.0        406    5,323,873      56,556.1  cudaLaunchKernel
      0.3        5,715,680          2   2,857,840.0  2,857,840.0  1,084,372    4,631,308   2,508,062.5  cudaFreeHost
      0.2        3,831,623          2   1,915,811.5  1,915,811.5    751,009    3,080,614   1,647,279.5  cudaMallocHost
      0.2        3,335,479          4     833,869.8    341,744.0     15,454    2,636,537   1,227,015.3  cudaFree
      0.1        2,183,944          1   2,183,944.0  2,183,944.0  2,183,944    2,183,944           0.0  cudaGraphInstantiate_v12000
      0.1          985,431         23      42,844.8     11,528.0      8,311      285,929      73,628.4  cudaMalloc
      0.0          703,349          4     175,837.3    165,577.5    143,599      228,595      37,215.7  cudaGraphDestroy_v10000
      0.0          458,251          1     458,251.0    458,251.0    458,251      458,251           0.0  cudaGraphExecDestroy_v10000
      0.0          426,714          4     106,678.5     86,223.5     54,426      199,841      63,897.5  cudaGraphLaunch_v10000
      0.0          396,709          4      99,177.3    105,130.0     79,940      106,509      12,846.8  cudaGraphExecUpdate_v10020
      0.0          319,325          4      79,831.3     77,259.0     68,726       96,081      12,957.9  cudaStreamEndCapture_v10000
      0.0          257,773        121       2,130.4      1,635.0        278       44,094       3,967.7  cudaMemsetAsync
      0.0          129,629          1     129,629.0    129,629.0    129,629      129,629           0.0  cuMemUnmap
      0.0          100,748          2      50,374.0     50,374.0     38,983       61,765      16,109.3  cuMemSetAccess
      0.0           58,903         36       1,636.2      1,028.5        919       19,411       3,096.0  cudaMemset
      0.0           48,085          2      24,042.5     24,042.5     18,130       29,955       8,361.5  cuMemCreate
      0.0           39,106          4       9,776.5      7,192.0      4,397       20,325       7,157.6  cudaStreamBeginCapture_v10000
      0.0           20,448          1      20,448.0     20,448.0     20,448       20,448           0.0  cudaStreamCreateWithFlags
      0.0            9,726          1       9,726.0      9,726.0      9,726        9,726           0.0  cuMemAddressReserve
      0.0            6,780          1       6,780.0      6,780.0      6,780        6,780           0.0  cudaStreamDestroy
      0.0            5,343          1       5,343.0      5,343.0      5,343        5,343           0.0  cuMemAddressFree
      0.0            5,062          2       2,531.0      2,531.0      2,246        2,816         403.1  cuMemMap
      0.0            1,832          1       1,832.0      1,832.0      1,832        1,832           0.0  cuMemGetAllocationGranularity
      0.0              682          2         341.0        341.0        123          559         308.3  cuMemRelease
      0.0              604          1         604.0        604.0        604          604           0.0  cuModuleGetLoadingMode
```


$ nsys stats --report cuda_api_sum /tmp/nsys-report-dc9b.nsys-rep | grep -i 'malloc\|memcpy'
     75.8    1,337,821,503        654   2,045,598.6      5,417.0      1,161  146,842,944   6,748,645.3  cudaMemcpyAsync
      0.2        3,831,623          2   1,915,811.5  1,915,811.5    751,009    3,080,614   1,647,279.5  cudaMallocHost
      0.1          985,431         23      42,844.8     11,528.0      8,311      285,929      73,628.4  cudaMalloc

$ nsys stats --report cuda_api_sum llama_profile.nsys-rep | grep -i 'malloc\|memcpy'
     75.9    1,335,230,491        654   2,041,636.8      5,297.5      1,246  146,747,384   6,741,506.2  cudaMemcpyAsync
      0.2        3,837,754          2   1,918,877.0  1,918,877.0    727,613    3,110,141   1,684,701.7  cudaMallocHost
      0.0          847,870         23      36,863.9     11,326.0      8,071      289,543      66,279.5  cudaMalloc 

