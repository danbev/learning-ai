### output_ids
This is a field/member of llama_context. The index into output_ids corresponds
to a token's index in the original llama_batch that the user provided.

```c++
struct llama_context {
    ...
    std::vector<int32_t> output_ids; // map batch token positions to ids of the logits and embd buffers
```
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
Notice that this will set/reset all the values in output_ids to -1.

So the value in this list is the position of that tokens result within the
contiguous logits and embd buffers. If the value is -1 it means that no output
was requested for that token.

For example, if we had a batch with 2 sequences, each with 2 tokens like this:
```
Batch contents:
  n_tokens: 4
  token[0]: tok=1    , pos=0, n_seq_id=1, seq_ids=[0], logits=0
  token[1]: tok=15043, pos=1, n_seq_id=1, seq_ids=[0], logits=1
  token[2]: tok=1    , pos=0, n_seq_id=1, seq_ids=[1], logits=0
  token[3]: tok=3834 , pos=1, n_seq_id=1, seq_ids=[1], logits=1
```
We would have a output_ids list like this:
```
[-1, 2, -1, 5] 
```
In this case token 0 does not have any output, token 1's output is at index 2
in the output buffer, token 2 does not have any output, and token 3's output is
at index 5 in the output buffer.

Now, when a decode is performed the batch will be split into smaller micro
batches and in the process.
```c++
llama_ubatch llama_batch_allocr::split_simple(uint32_t n_ubatch) {
    // find the first unused token
    uint32_t cur_idx = 0;
    while (cur_idx < used.size() && used[cur_idx]) {
        ++cur_idx;
    }

    // we are done
    if (cur_idx >= used.size()) {
        return {};
    }

    std::vector<int32_t> idxs;

    while (true) {
        idxs.push_back(cur_idx);

        used[cur_idx] = true;
        ++n_used;

        ++cur_idx;

        if (cur_idx >= used.size()) {
            break;
        }

        if (idxs.size() >= n_ubatch) {
            break;
        }
    }

    return ubatch_add(idxs, idxs.size(), false);
}
```

```console
503	    return ubatch_add(idxs, idxs.size(), false);
(gdb) p idxs
$29 = std::vector of length 6, capacity 8 = {0, 1, 2, 3, 4, 5}
```
```c++
llama_ubatch llama_batch_allocr::ubatch_add(const std::vector<int32_t> & idxs, uint32_t n_seqs, bool equal_seqs) {
    ...

        if (udata->output[i]) {
            out_ids.push_back(idxs[i]);
        }
}
```
So for each tokens that is marked as an output we store its original index in
the out_ids list:
```console
(gdb) p out_ids
$33 = std::vector of length 0, capacity 0
(gdb) p idxs
$34 = std::vector of length 6, capacity 8 = {0, 1, 2, 3, 4, 5}
(gdb) p i
$35 = 5

(gdb) n
(gdb) p out_ids
$36 = std::vector of length 1, capacity 1 = {5}
```

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
$37 = std::vector of length 1, capacity 1 = {5}
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
And recall that output_ids are all -1 at this stage. And we are going to set
the out_id position which is:
```console
(gdb) p out_id
$39 = 5
```
This is the token at index 5 in our original batch. So we are setting this
as output_ids[5] = 0.
And we also set sorted_output to false since out_id (5) is not equal to i (0).

So then we will enter the following block:
```c++
        if (!sorted_output) {
            for (uint32_t i = 0; i < n_outputs - 1; ++i) {
                uint32_t j_min = i;
                for (uint32_t j = i + 1; j < n_outputs; ++j) {
                    if (out_ids[j] < out_ids[j_min]) {
                        j_min = j;
                    }
                }
                if (j_min == i) {
                    continue;
                }
                std::swap(out_ids[i], out_ids[j_min]);

                // remember the swaps and apply them lazily upon logits/embeddings access
                output_swaps.push_back({ i, j_min });
            }

            std::fill(output_ids.begin(), output_ids.end(), -1);

            for (uint32_t i = 0; i < n_outputs; ++i) {
                output_ids[out_ids[i]] = i;
            }
        }
```
And this will iterate over all the outputs - 1. In this particular case we only
have one output so the output_ids are reset back to -1 again:
And then we iterate over all the outputs again:
```c++
            for (uint32_t i = 0; i < n_outputs; ++i) {
                output_ids[out_ids[i]] = i;
            }
```
This time we are setting output_ids[5] = 0 again.
```console
$50 = std::vector of length 2048, capacity 2048 = {-1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1
```
So if a user now requests the logits for token index 5 in the original batch, we
can use the output_ids[5] which is 0, to know that this data is at index 0 in
the logits pinned memory.

And notice that the async copy is done prior to this step:
```c++
        // extract logits
        if (t_logits && n_outputs > 0) {
            ggml_backend_t backend_res = ggml_backend_sched_get_tensor_backend(sched.get(), t_logits);

            float * logits_out = logits + n_outputs_prev*n_vocab;

            if (n_outputs) {
                ggml_backend_tensor_get_async(backend_res, t_logits, logits_out, 0, n_outputs*n_vocab*sizeof(float));
            }
        }
```
Here, logits_out is pointing to the start of the logits buffer for this batch.
And logits is the start of our pinned memory buffer. So we will copy the tensor
data from the GPU that is in t_logits to the pinned memory location pointed to
by logits_out.
