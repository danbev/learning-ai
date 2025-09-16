## batch/ubatch/sbatch

Note that a new ubatch is created for each iteration of this loop!
And, notice `split_simple` is a function/method on sbatch, so lets take a look
at the state of this instance:
```console
(gdb) ptype *this
type = struct llama_sbatch {
    size_t n_tokens;
    size_t n_embd;
    bool logits_all;
    std::vector<unsigned long> ids;
    std::vector<unsigned long> out_ids;
    std::vector<llama_sbatch_seq> seq;
    const llama_batch *batch;
    std::vector<int> ubatch_token;
    std::vector<float> ubatch_embd;
    std::vector<int> ubatch_pos;
    std::vector<int> ubatch_n_seq_id;
    std::vector<int*> ubatch_seq_id;
    std::vector<signed char> ubatch_output;

    llama_ubatch reserve_ubatch(size_t, bool);
    void add_seq_to_ubatch(llama_ubatch &, llama_sbatch_seq &, size_t);
    llama_ubatch split_simple(size_t);
    llama_ubatch split_equal(size_t);
    llama_ubatch split_seq(size_t);
    void from_batch(const llama_batch &, size_t, bool, bool);
}

(gdb) p *this
$20 = {
n_tokens = 73,
n_embd = 4096,
logits_all = false,
ids = std::vector of length 73, capacity 73 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72},
out_ids = std::vector of length 0, capacity 0,
seq = std::vector of length 1, capacity 1 = {{ n_seq_id = 0, seq_id = 0x0, offset = 0, length = 73, all_seq_id = 0}},
batch = 0x7fffffffd670,
ubatch_token = std::vector of length 0, capacity 0,
ubatch_embd = std::vector of length 0, capacity 0,
ubatch_pos = std::vector of length 0, capacity 0,
ubatch_n_seq_id = std::vector of length 0, capacity 0,
ubatch_seq_id = std::vector of length 0, capacity 0,
ubatch_output = std::vector of length 0, capacity 0}
```
So an sbatch is initialized in `llama_decode_internal` by using information for
the batch that was passed in. This will keep track of the number of tokens that
have been processed and will be used to create ubatches. 

In this case `n_ubatch` is 32.
```c++
    // simple split, unknown number of sequences of unequal lengths
    llama_ubatch split_simple(size_t n_ubatch) {
        n_ubatch = n_tokens < n_ubatch ? n_tokens : n_ubatch;
        llama_ubatch ubatch = reserve_ubatch(n_ubatch, /* has_embd */ batch->embd != nullptr);
        ubatch.equal_seqs = false;
        if (!seq.empty()) {
            llama_sbatch_seq & s = seq[0];
            size_t length = s.length < n_ubatch ? s.length : n_ubatch;
            GGML_ASSERT(seq.size() == 1 && s.n_seq_id == 0); // don't mix with other splits
            add_seq_to_ubatch(ubatch, s, length);
        }
        return ubatch;
    }
```
This time (compared to the previous example where the sequences/propts were much
smaller) the `n_ubatch` will be changed from 72 to 32 (just the local variable
that is).
```console
(gdb) p n_ubatch
$18 = 32
```
In `add_seq_to_ubatch` we then have:
```c++
                ubatch.token = batch->token + seq.offset;
```
This is using the `llama_batch` token data and the offset is 0 in this case.
```c++
        ...

        if (ubatch.n_tokens == 0 && ubatch.n_seqs == 0) {
            ubatch.n_seq_tokens = ubatch.equal_seqs ? length : 1;
        }
        ubatch.n_tokens += length;
        ubatch.n_seqs += ubatch.equal_seqs ? 1 : length; // virtual sequences for simple splits
        seq.offset += length;
        seq.length -= length;
        n_tokens -= length;
```
So the ubatch `n_tokens` will be updated to 32 and notice that `seq` which is
a reference passed in will be updated to keep track of the number of tokens
that have been added to the updata batch but incrementing the offset, and
decrementing the lenght which is currently 73.
Also notice that the sbatch `n_tokens` will also be decremented and recall
that this is the field that is used in the while loop in
`llama_decode_internal`.

```c++
struct llama_sbatch_seq {
    int32_t n_seq_id;
    llama_seq_id * seq_id;
    size_t offset;
    size_t length;

    // helper for smoother batch API transition -- can be deprecated in the future
    llama_seq_id all_seq_id; // used if seq_id == NULL
};
```
Now, lets take a look at the second iteration of the while loop.
```c++
            ubatch = lctx.sbatch.split_simple(n_ubatch);
```
This time around `n_tokens` is 41:
```console
(gdb) p n_tokens
$33 = 41
(gdb) p length
$34 = 32
(gdb) p seq.offset
$67 = 32
```
So this time when the updates take place in `add_seq_to_ubatch` the offset will
be an offset into the original `llama_batch`:
```c++
                ubatch.token = batch->token + seq.offset;
```
The last things to happen in `add_seq_to_ubatch` are:
```c++
        ubatch.n_tokens += length;
        ubatch.n_seqs += ubatch.equal_seqs ? 1 : length; // virtual sequences for simple splits
```
```console
(gdb) p ubatch.n_tokens
$75 = 32
(gdb) p ubatch.n_seqs
$74 = 32
```
For some reason I'm not understanding what `n_seqs` is supposed to represent and
this might be because I've only stepped through an example that uses the
simple split. TODO: try a recurrent model with the same setting and see if this
makes more sense then. To me this field seems to specify the number of tokens
but there already is a field for that `n_tokens` so I'm not sure what this
field is for.

Then we have the updating of the `seq` reference, and `n_tokens` field of
`sbatch`:
```c++
        seq.offset += length;
        seq.length -= length;
        n_tokens -= length;
        GGML_ASSERT(ubatch.n_tokens == ubatch.n_seq_tokens * ubatch.n_seqs);
```
The value of `seq.offset` before the update is:
```console
(gdb) p seq.offset
$40 = 32
(gdb) p seq.length
$79 = 41
(gdb) p this.n_tokens
$81 = 41
```
And after the updates:
```console
(gdb) p seq.offset
$44 = 64
(gdb) p seq.length
$84 = 9
(gdb) p this.n_tokens
$83 = 9
```


Now, lets try the same example but using a Mamba model which is a recurrent
model to see the difference and perhaps gain a better understanding of the
sbatch/ubatch fields in the process.
Using `mamba-1.4b-f16.gguf` we can see that `simple_split` is now false when
calling `llama_sbatch::from_batch`. This will then cause the following block
to be executed. And lets first inspect ids which will be used in this block:
```console
(gdb) p ids
$5 = std::vector of length 64, capacity 64 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
  22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
  52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63}
```

```c++
        std::sort(ids.begin(), ids.end(),
            [&batch](size_t a, size_t b) {
                int32_t n_seq_a = batch.n_seq_id ? batch.n_seq_id[a] : 1;
                int32_t n_seq_b = batch.n_seq_id ? batch.n_seq_id[b] : 1;
                // sort by seq_id, then by pos
                if (n_seq_a == n_seq_b) {
                    if (batch.seq_id) {
                        for (int32_t i = 0; i < n_seq_a; ++i) {
                            llama_seq_id seq_id_a = batch.seq_id[a][i];
                            llama_seq_id seq_id_b = batch.seq_id[b][i];
                            // smaller seq_ids go first
                            if (seq_id_a != seq_id_b) {
                                return seq_id_a < seq_id_b;
                            }
                        }
                    }
                    // when all else is equal, sort by pos
                    if (batch.pos) {
                        return batch.pos[a] < batch.pos[b];
                    }
                    // no pos, sort by id (assuming batch.all_pos_1 is positive)
                    return a < b;
                }
                // shared prompts go first
                return n_seq_a > n_seq_b;
            }
        );
```
So this will sort the ids vector using the custom lambda function passing in
a referece to the `llama_batch`. This is first checking if the number of
sequence ids that a tokens have are the same and if that is not the case the
ones with more tokens that belong to more sequences will come before others, the
lambda will return true in this case. If both tokens have the same number of 
sequences they belong to, then the above will iterate over all the sequences
that `a` belongs to. If these differ that is either the first token has tokens
that belong to other sequences, of the second token has tokens that belong to
other sequences then the one with the smaller sequence id will come first.
Else we will sort by the position of the tokens in the batch. And if there are
not positions then we sort by token id.
Note that this is an inplace sort so the `ids` vector will be sorted after this
call to sort.
This will do nothing for us as we have not set/configured any tokens so the
belong to multiple sequences:
```console
(gdb) p ids
$23 = std::vector of length 64, capacity 64 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
  21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
  51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63}
```
Lets try updating the batch example to see how this may work.

```c++
        // init seq
        llama_sbatch_seq * last_seq = nullptr;

        if (batch.n_seq_id != nullptr && batch.seq_id != nullptr) {
            for (size_t i = 0; i < n_tokens; ++i) {
                const size_t bi = ids[i];
                const int32_t n_seqs = batch.n_seq_id[bi];
                llama_seq_id * seq_ids = batch.seq_id[bi];
                if (last_seq != nullptr) {
                    bool same = n_seqs == last_seq->n_seq_id;
                    for (int32_t j = 0; same && j < n_seqs; ++j) {
                        if (seq_ids[j] != last_seq->seq_id[j]) {
                            same = false;
                        }
                    }
                    if (same) {
                        last_seq->length += 1;
                        continue;
                    }
                }
                llama_sbatch_seq new_seq = {n_seqs, seq_ids, i, 1, batch.all_seq_id};
                seq.push_back(new_seq);
                last_seq = &seq.back();
            }
        } else {
            llama_sbatch_seq new_seq = {1, nullptr, 0, n_tokens, batch.all_seq_id};
            seq.push_back(new_seq);
        }
```
Lets got through the first token in this case:
```console
(lldb) p ids[0]
(std::vector<unsigned long>::value_type) 1

(lldb) p batch.n_seq_id[1]
(int32_t) 2

(lldb) p last_seq
(llama_sbatch_seq *) nullptr
```
So in this case there is now `last_seq` and it will be populated by:
```c++
                llama_sbatch_seq new_seq = {n_seqs, seq_ids, i, 1, batch.all_seq_id};
                seq.push_back(new_seq);
                last_seq = &seq.back();
```
This will create a new `llama_sbatch_seq` with 2 as the number of sequences, the pointer
to the sequenced is that this tokens has, a length of 1 (initially):
```console
(lldb) p new_seq
(llama_sbatch_seq) {
  n_seq_id = 2
  seq_id = 0x00006000032257d0
  offset = 0
  length = 1
  all_seq_id = 0
}
``
And then add it to the `seq` vector and then set `last_seq` to point to the last element
in the vector. 

```c++
struct llama_sbatch_seq {
    int32_t n_seq_id;
    llama_seq_id * seq_id;
    size_t offset;
    size_t length;

    // helper for smoother batch API transition -- can be deprecated in the future
    llama_seq_id all_seq_id; // used if seq_id == NULL
};
```

_wip_

### batch/ubatch/sbatch
So a batch, or rather a `llama_batch` is what we pass into the `llama_decode`
function and is really the only thing that an external caller knows anything
about (well that is not entirely true as they can set a ubatch value but the
as a command line argumument `-ub/--ubatch-size`).
The internal decode operation is/can be split into smaller units called ubatches.
The internal decode will create an sbatch, sequence-aware that manages the ubatches. I
was not sure what this actually meant, the sequence-aware batch and there was a
disussion where this was also asked where I replied with the following:
```. 
sbatch stands for sequence-aware batch which can be found from this comment. My understanding of what this means is that for recurrent models they can benefit from processing batches with sequences of equal length. The way this works is that if there are multiple sequences in a batch (llama_batch that is passed to llama_decode) they will be split into ubatches of equal sequence length.
The sbatch will manage this (hence the name sequence aware batch) and produce an ubatch which will then be used to build the computation graph, set the inputs, and then compute the graph (the forward pass). So the ubatch is what is actually passed to the model and I think it stands for micro batch.
This process will then continue by handling the remaining tokens for the sequences, again making sure that they are of equal size.

There are some more details in the following comment which also contains an example:
#7531 (comment)
```
