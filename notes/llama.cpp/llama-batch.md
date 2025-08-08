## llama_batch
_wip_

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
