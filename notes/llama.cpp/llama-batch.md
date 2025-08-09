## llama_batch
_wip_

### batch
```console
(lldb) expr batch
(llama_batch) $4 = {
  n_tokens = 10
  token = 0x0000000144abb400
  embd = 0x0000000000000000
  pos = 0x0000000144cac400
  n_seq_id = 0x0000000144cacc00
  seq_id = 0x0000000144cad400
  logits = 0x0000000143187990
}
```
```c++
struct llama_batch {
    int32_t      n_tokens;    // Total number of tokens
    llama_token* token;       // Array of token IDs
    float*       embd;        // Embeddings (if provided instead of tokens)
    llama_pos*   pos;         // Position of each token in its sequence
    int32_t*     n_seq_id;    // Number of sequences each token belongs to
    llama_seq_id** seq_id;    // For each token, array of sequence IDs it belongs to
    int8_t*      logits;      // Whether to compute logits for each token
}
```
So we can see that a token in the batch can belong to multiple sequences, if its `n_seq_id` is greater
than 1. In this case, the `seq_id` array will contain multiple sequence IDs for that token.

```console
Batch with 6 tokens:
Token 0,1,2 → sequence 0
Token 3,4,5 → sequence 1

n_tokens = 6
token = [tok0, tok1, tok2, tok3, tok4, tok5]
n_seq_id = [1, 1, 1, 1, 1, 1]  // Each token belongs to 1 sequence
seq_id[0] → [0]  // Token 0 belongs to sequence 0
seq_id[1] → [0]  // Token 1 belongs to sequence 0
seq_id[2] → [0]  // Token 2 belongs to sequence 0
seq_id[3] → [1]  // Token 3 belongs to sequence 1
seq_id[4] → [1]  // Token 4 belongs to sequence 1
seq_id[5] → [1]  // Token 5 belongs to sequence 1
```

### ubatch
```console
(lldb) type lookup llama_ubatch
struct llama_ubatch {
    uint32_t b_equal_seqs;
    uint32_t n_tokens;
    uint32_t n_seq_tokens;
    uint32_t n_seqs;
    uint32_t n_seqs_unq;
    llama_token *token;
    float *embd;
    llama_pos *pos;
    int32_t *n_seq_id;
    llama_seq_id **seq_id;
    llama_seq_id *seq_id_unq;
    int32_t *seq_idx;
    int8_t *output;
    struct data_t;
    std::shared_ptr<llama_ubatch::data_t> data;
    bool equal_seqs() const;
}
```
```c++
struct llama_ubatch {
    uint32_t b_equal_seqs;    // Whether all tokens have same sequence pattern (split equal)
    uint32_t n_tokens;        // Total tokens (same as batch)
    uint32_t n_seq_tokens;    // Tokens per sequence
    uint32_t n_seqs;          // Number of sequence sets in this ubatch
    uint32_t n_seqs_unq;      // Number of unique sequence IDs

    // Per-token data (same as batch)
    llama_token* token;       // Token IDs
    float*       embd;        // Embeddings
    llama_pos*   pos;         // Positions
    int32_t*     n_seq_id;    // Number of sequences per token
    llama_seq_id** seq_id;    // Sequence IDs per token (pointers to original)

    llama_seq_id* seq_id_unq; // Compact list of unique seq IDs [0, 1]
    int32_t*      seq_idx;    // Maps seq_id → index in seq_id_unq

    int8_t*      output;      // Replaces batch.logits

    // Internal data
    std::shared_ptr<data_t> data;  // Shared data structures
}
```

```console
Batch with 6 tokens:
Token 0,1,2 → sequence 0
Token 3,4,5 → sequence 1

b_equal_seqs = 1        // All tokens follow same pattern (3 per seq)
n_seq_tokens = 3        // 3 tokens per sequence
n_seqs = 2              // 2 sequence sets in batch
n_seqs_unq = 2          // 2 unique sequence IDs

seq_id_unq = [0, 1]     // List of unique sequences
seq_idx = [0, 1, -1, -1, ...]  // Remapping table
         // ↑  ↑
         // seq 0 at index 0
         //    seq 1 at index 1

// Output control (replaces logits)
output = [0, 0, 1, 0, 0, 1]  // Which tokens generate output
```

#### `seq_id_unq` and `seq_idx`
Instead of having to have an array of 64 entries, one for each possible sequence if we have
less then 64 only the actuall uses sequence ids are stored in `seq_id_unq`. And `seq_idx` is used
to identify which of these a token belongs to.

```
// Only allocate for ACTIVE sequences
float kv_cache_compact[2][max_tokens][dim];  // Just 2 sequences!

// Remapping via seq_idx:
seq_id_unq = [5, 27]        // The actual sequence IDs
seq_idx[5] = 0              // Sequence 5 → use kv_cache_compact[0]
seq_idx[27] = 1             // Sequence 27 → use kv_cache_compact[1]

// Now fully utilized:
kv_cache_compact[0][...]    // USED for sequence 5
kv_cache_compact[1][...]    // USED for sequence 27

// Memory usage: 2 × max_tokens × dim × sizeof(float)
// 100% of allocated memory is used!
```

So without this compaction we would have something like this:
```c++
int token_idx = 2;
llama_seq_id seq_id = *ub.seq_id[token_idx];  // = 27

// float* kv = kv_cache[27];  // Direct index, but cache is huge

int compact_idx = ub.seq_idx[seq_id];  // seq_idx[27] = 1
float* kv = kv_cache_compact[compact_idx];  // Use slot 1
```
So this will allow the kv-cache and attention mask matrices to be much smaller, so
instead of having to reserve 64 entires for the kv-cache we can just reserve
the number of sequences that are actually used in the batch.

So we use the token id to look up the sequence id for that particular token using
`ubatch.seq_id(token_id)`. This will give us a sequence id for the token. We then
use this sequence id to look up the index of the sequence in the `seq_id_unq` array
using `ubatch.seq_idx(seq_id)`. This will give us the index of the sequence in the
`seq_id_unq` array.
So something like this:
```c++
(lldb) p ubatch.token[3]
(llama_token) 1                // The actual token value (vocab ID)

(lldb) p ubatch.n_seq_id[3]
(int32_t) 1                    // Number of sequences this token belongs to

(lldb) p ubatch.seq_id[3][0]  // Token at index 3 belongs to sequence 1, only 1 hence [0]
(llama_seq_id) 1

(lldb) p ubatch.seq_idx[1]   // Sequence 1 is at compact index 1
(int32_t) 1

(lldb) p ubatch.seq_id_unq[1] // Verify: compact slot 1 contains sequence 1
(llama_seq_id) 1
```

### Walkthrough of non-unified kv-cache for llama_batch
First we need to make sure that the environment variable `LLAMA_SET_ROWS` is set to `1`:
```console
$ export LLAMA_SET_ROWS=1
```
And then start the example program `simple-prompt-multi`which sets up a batch with two
sequences:
```console
$ lldb simple-prompt-multi
```
And well start by setting a breakpoint in `llama_batch_allocr::init_batch`:
```console
(lldb) br set -f llama-kv-cache-unified.cpp -l 485
(lldb) r
```

The entry point of this this session will be `llama_context::decode`:
```c++
    if (!balloc->init(batch_inp, vocab, memory.get(), n_embd,
        cparams.kv_unified ? LLAMA_MAX_SEQ : cparams.n_seq_max, output_all)) {
```
In this case we are looking at the non-unified case so `cparams.kv_unified` will be `false`:
```console
(lldb) expr  cparams.kv_unified
(bool) $1 = false
```
And `balloc` is of type `llama_batch_allocr`:
```
(lldb) expr this->balloc
(std::unique_ptr<llama_batch_allocr>) $3 = llama_batch_allocr @ 0x00000001431056b0 {
  pointer = 0x00000001431056b0
}
(lldb) type lookup llama_batch_allocr
class llama_batch_allocr {
    llama_batch batch;
    const llama_vocab *vocab;
    const uint32_t n_pos_per_embd;
    uint32_t n_embd;
    uint32_t n_seq_max;
    uint32_t n_outputs;
    std::array<int, 1> seq_id_0;
    std::vector<int> pos;
    std::vector<int> n_seq_id;
    std::vector<int *> seq_id;
    std::vector<int> seq_id_unq;
    std::vector<int> seq_idx;
    std::vector<signed char> output;
    bool has_cpl;
    std::vector<std::set<int> > seq_pos;
    std::vector<std::vector<bool> > seq_cpl;
    std::vector<std::bitset<64> > seq_set;
    std::unordered_map<std::bitset<64>, std::vector<int> > seq_set_map;
    std::vector<int> out_ids;
    uint32_t n_used;
    std::vector<bool> used;
    int debug;
public:
    llama_batch_allocr(uint32_t);
    bool init(const llama_batch &, const llama_vocab &, const llama_memory_i *, uint32_t, uint32_t, bool);
    const llama_batch &get_batch() const;
    uint32_t get_n_tokens() const;
    uint32_t get_n_outputs() const;
    uint32_t get_n_used() const;
    std::vector<int> &get_out_ids();
    llama_pos seq_pos_min(llama_seq_id) const;
    llama_pos seq_pos_max(llama_seq_id) const;
    void split_reset();
    llama_ubatch split_simple(uint32_t);
    llama_ubatch split_equal(uint32_t, bool);
    llama_ubatch split_seq(uint32_t);
    llama_ubatch ubatch_reserve(uint32_t, uint32_t);
    void clear();
    llama_ubatch ubatch_add(const std::vector<int> &, uint32_t, bool);
    void ubatch_print(const llama_ubatch &, int);
}
```

So the first thing that happens is that `llama_batch_allocr::init` is called:
```c++
bool llama_batch_allocr::init(
        const llama_batch & batch_inp,
        const llama_vocab & vocab,
        const llama_memory_i * memory,
        uint32_t n_embd,
        uint32_t n_seq_max,
        bool output_all) {
    ...
    batch = batch_inp;
}
```
And our input batch looks like this:
```console
(lldb) p batch_inp
(const llama_batch &) 0x000000016fdfe698: {
  n_tokens = 10
  token = 0x0000000144abb400
  embd = 0x0000000000000000
  pos = 0x0000000144cac400
  n_seq_id = 0x0000000144cacc00
  seq_id = 0x0000000144cad400
  logits = 0x0000000143187990
}
```
There are a number of sanity checks that are performed in init and also it will add missing
information if information is missing from the batch, and notice that balloc will store a reference
to `batch_inp`.
In this case all the fields are set so I'll skip this part.
```c++
    //
    // compute stats
    //

    this->n_embd    = n_embd;
    this->n_seq_max = n_seq_max;

    // count the outputs in this batch
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        n_outputs += batch.logits[i] != 0;
    }
```
Notice how the the number of outputs is counted here, it checks which tokens have there logits
set and count them.

Next, there will be a check to see if there are sequences that are coupled
```
    has_cpl = false;

    // determine coupled sequences
    // these are pairs of sequences that have at least one token in the input batch that is assigned to both of them
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        const llama_seq_id s0 = batch.seq_id[i][0];

        for (int32_t s = 0; s < batch.n_seq_id[i]; ++s) {
            const llama_seq_id s1 = batch.seq_id[i][s];

            seq_pos[s1].insert(batch.pos[i]);

            if (s > 0) {
                // mark that sequence s1 is coupled to s0
                seq_cpl[s1][s0] = true;

                // note: tracking the other way around is not necessary for now
                //seq_cpl[s0][s1] = true;

                has_cpl = true;
            }
        }
    }
```


One thing to note when we use a non-unified kv-cache is how ubatches are handled:
```c++
llama_memory_context_ptr llama_kv_cache_unified::init_batch(
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
In this case `n_streams` will not be `1` and we will use the `split_equal`:
```console
(gdb) p n_stream
$55 = 2
```
Now, this is creating a microbatch where all sequences have the same number of
tokens. Notice that there is a vector of `llama_ubatch` which will be added
to during the loop.

This is what our original batch looked like:
```console
(gdb) p batch
$56 = {n_tokens = 10, token = 0x55555562bf00,
embd = 0x0, pos = 0x555555638af0, n_seq_id = 0x555555639300,
seq_id = 0x555555639b10, logits = 0x55555563cce0 ""}

(gdb) p batch.token[0]
$57 = 1
(gdb) p batch.token[1]
$58 = 15043
(gdb) p batch.token[2]
$59 = 29871
(gdb) p batch.token[3]
$60 = 1
(gdb) p batch.token[4]
$61 = 3951
(gdb) p batch.token[5]
$62 = 12355
(gdb) p batch.token[6]
$63 = 267
(gdb) p batch.token[7]
$64 = 14890
(gdb) p batch.token[8]
$65 = 907
(gdb) p batch.token[9]
$66 = 314

(gdb) p this.model.vocab.token_get_text(1)
$70 = 0x555555643808 "<s>"
(gdb) p this.model.vocab.token_get_text(2)
$71 = 0x555555643830 "</s>"
(gdb) p this.model.vocab.token_get_text(1)
$72 = 0x555555643808 "<s>"
(gdb) p this.model.vocab.token_get_text(15043)
$73 = 0x5555556d6658 "▁Hello"
(gdb) p this.model.vocab.token_get_text(29871)
$74 = 0x555555767338 "▁"
(gdb) p this.model.vocab.token_get_text(1)
$75 = 0x555555643808 "<s>"
(gdb) p this.model.vocab.token_get_text(3951)
$76 = 0x55555566a138 "▁Dan"
(gdb) p this.model.vocab.token_get_text(12355)
$77 = 0x5555556bc258 "▁lov"
(gdb) p this.model.vocab.token_get_text(267)
$78 = 0x555555646198 "es"
(gdb) p this.model.vocab.token_get_text(14890)
$79 = 0x5555556d4e70 "▁ice"
(gdb) p this.model.vocab.token_get_text(907)
$80 = 0x55555564c598 "▁cre"
(gdb) p this.model.vocab.token_get_text(314)
$81 = 0x5555556468f0 "am"
```
Now, we know we have two sequences, the first one has 3 tokens and the second
one has 7. 

```console
(gdb) p batch
$36 = {n_tokens = 10, token = 0x555555638af0, embd = 0x0,
pos = 0x555555639b00, n_seq_id = 0x55555563ab10, seq_id = 0x55555563bb20,
logits = 0x55555557b1c0 ""}
(gdb) p *batch.seq_id[1]
$39 = 0
(gdb) p *batch.seq_id[2]
$40 = 0
(gdb) p *batch.seq_id[3]
$41 = 1
(gdb) p *batch.seq_id[4]
$42 = 1
(gdb) p *batch.seq_id[5]
$43 = 1
(gdb) p *batch.seq_id[6]
$44 = 1
(gdb) p *batch.seq_id[7]
$45 = 1
(gdb) p *batch.seq_id[8]
$46 = 1
(gdb) p *batch.seq_id[9]
$47 = 1
```
In llama_batch there is a `seq_set` which is a vector of bitsets that tell us
which sequence a token belongs to:
```console
(gdb) p seq_set[0]
(gdb) p seq_set[0]
$67 = std::bitset = {[0] = 1}
```
So we are asking here about token 0, and it belongs to sequence 0 because that
bit is set:
`[0] = 1` means bit 0 (sequence 0) is set/active.

```console
$49 = std::vector of length 10, capacity 16 = {
std::bitset = {[0] = 1},           // [0] = 1 means bit 0 (sequence 0) is set/active
std::bitset = {[0] = 1},           //  ↑
std::bitset = {[0] = 1},           //  bit 0
std::bitset = {[1] = 1},           // [1] = 1 means bit 1 (sequence 1) is set/active
std::bitset = {[1] = 1},           //  ↑
std::bitset = {[1] = 1},           //  bit 1 (sequence 1) is set/active
std::bitset = {[1] = 1},
std::bitset = {[1] = 1},
std::bitset = {[1] = 1},
std::bitset = {[1] = 1}}
```
So that can tell us which sequence a token belongs to, and with the information
we can get all the tokens that belong to that sequence:
```console
(gdb) p seq_set_map
$50 = std::unordered_map with 2 elements = {
[std::bitset = {[0] = 1}] = std::vector of length 3, capacity 4 = {0, 1, 2}}
[std::bitset = {[1] = 1}] = std::vector of length 7, capacity 8 = {3, 4, 5, 6, 7, 8, 9},
```
These structures allows us to quickly look up which sequence a token belongs to:
```console
(gdb) p seq_set_map[seq_set[0]]
$56 = std::vector of length 3, capacity 4 = {0, 1, 2}
(gdb) p seq_set_map[seq_set[1]]
$57 = std::vector of length 3, capacity 4 = {0, 1, 2}
(gdb) p seq_set_map[seq_set[2]]
$58 = std::vector of length 3, capacity 4 = {0, 1, 2}
(gdb) p seq_set_map[seq_set[3]]
$59 = std::vector of length 7, capacity 8 = {3, 4, 5, 6, 7, 8, 9}
(gdb) p seq_set_map[seq_set[4]]
$60 = std::vector of length 7, capacity 8 = {3, 4, 5, 6, 7, 8, 9}
(gdb) p seq_set_map[seq_set[5]]
$61 = std::vector of length 7, capacity 8 = {3, 4, 5, 6, 7, 8, 9}
(gdb) p seq_set_map[seq_set[6]]
$62 = std::vector of length 7, capacity 8 = {3, 4, 5, 6, 7, 8, 9}
(gdb) p seq_set_map[seq_set[8]]
$63 = std::vector of length 7, capacity 8 = {3, 4, 5, 6, 7, 8, 9}
(gdb) p seq_set_map[seq_set[9]]
$64 = std::vector of length 7, capacity 8 = {3, 4, 5, 6, 7, 8, 9}
```

```console
(gdb) p *udata
$100 = {token = std::vector of length 6, capacity 6 = {1, 15043, 29871, 1, 3951, 12355}, embd = std::vector of length 0, capacity 0,
  pos = std::vector of length 6, capacity 6 = {0, 1, 2, 0, 1, 2}, n_seq_id = std::vector of length 6, capacity 6 = {1, 1, 1, 1, 1,
    1}, seq_id = std::vector of length 6, capacity 6 = {0x5555555bfa20, 0x555555637930, 0x555555637950, 0x555555637ac0,
    0x555555637a80, 0x5555555a2570}, seq_id_unq = std::vector of length 0, capacity 0,
  seq_idx = std::vector of length 64, capacity 64 = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, output = std::vector of length 6, capacity 6 = {0 '\000', 0 '\000', 1 '\001',
    0 '\000', 0 '\000', 0 '\000'}}
(gdb) p udata->token
$101 = std::vector of length 6, capacity 6 = {1, 15043, 29871, 1, 3951, 12355}
(gdb) p udata->token[0]
$102 = 1
(gdb) p udata->token[1]
$103 = 15043
(gdb) p udata->token[2]
$104 = 29871
(gdb) p udata->token[3]
$105 = 1
(gdb) p udata->token[4]
$106 = 3951
(gdb) p udata->token[5]
$107 = 12355
(gdb) p udata->seq_id[0]
$108 = (int *) 0x5555555bfa20
(gdb) p *udata->seq_id[0]
$109 = 0
(gdb) p *udata->seq_id[1]
$110 = 0
(gdb) p *udata->seq_id[2]
$111 = 0
(gdb) p *udata->seq_id[3]
$112 = 1
(gdb) p *udata->seq_id[4]
$113 = 1
(gdb) p *udata->seq_id[5]
$114 = 1
```

```console
Token Index: 0      1      2      3      4      5
Token ID:    1      15043  29871  1      3951   12355
Position:    0      1      2      0      1      2
Sequence:    0      0      0      1      1      1
```

Now, `seq_id_unq` is a compact list of unique sequence IDs actually used in this
ubatch. And `seq_idx` maps from sequence ID to an index in seq_id_unq.
```console
(gdb) ptype udata->seq_id_unq
type = std::vector<int>

(gdb) ptype udata->seq_id
type = std::vector<int*>
```
```c++
llama_ubatch llama_batch_allocr::ubatch_add(const std::vector<int32_t> & idxs, uint32_t n_seqs, bool equal_seqs) {
    ...

    for (uint32_t s = 0; s < n_seq_max; ++s) {
        if (seq_set_unq.test(s)) {
            udata->seq_idx[s] = udata->seq_id_unq.size();
            udata->seq_id_unq.push_back(s);
        }
    }
```
Now, what is happeing is that first the size of the `seq_id_unq` vector is used
for the index (this is before it is added to data->seq_is_unq on the following
line), so this allows the correct index to be added. And notice that this is
guarded by the if statement so there is no guarantee that s can act as a sequence
id.
So after that we have:
```console
(gdb) p udata->seq_id_unq
$127 = std::vector of length 2, capacity 2 = {0, 1}
```

And the final ubatch create will look like this:
```console
(gdb) p res
$129 = {b_equal_seqs = 1, n_tokens = 6, n_seq_tokens = 3, n_seqs = 2,
n_seqs_unq = 2, token = 0x555555dfb280, embd = 0x0, pos = 0x555555dfb300,
n_seq_id = 0x555555dfb320, seq_id = 0x55555557d1c0,
seq_id_unq = 0x555555dfb3a0, seq_idx = 0x5555555a2650, output = 0x555555dfb340 "",
data = std::shared_ptr<llama_ubatch::data_t> (use count 1, weak count 0) = {get() = 0x555555637820}}
```
