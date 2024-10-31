### Self-Extend/Grouped Attention options
This is for something called `self-extend` which is described in the following
paper: https://arxiv.org/pdf/2401.01325

When sequences exceed the context window size used during training, the model
encounters relative positions (think RoPE) it hasn't seen before, leading to
performance degradation. So during training the LLM was trained using a specific
context length, and the relative positions it learned where within this range.
During inference if the relative positions are larger than this range the
LLM will have out-of-distribution (O.O.D) problem.

Self-extend is a way to map these relative positions that are larger than the
context length that the model was trained on at _inference_ time.

These are the options related to Grouped Query Attention (gqa):
```console
$ ./llama-cli --help | grep group
  -gan,  --grp-attn-n N           group-attention factor (default: 1)
  -gaw,  --grp-attn-w N           group-attention width (default: 512.0)
```
`grp-attn-n` is the group-attention factor which is used in the floor operation
to divide the relative positions into groups. The default is 1 which means that
the relative positions are not divided into groups.

`grp-attn-w` specifies the total width of tokens used in self-extend. So
normally in the attention relative position "encoding/calculation" the positions
that are outside of the context length that the model was trained can cause
issues for the model because it was not trained on these positions. The limit
here is the models training context length. Here this limit is made configurable
so that it can be adjusted to other values than the models training context
length. This provides greater flexibility in how the model can be used.
So with this and the group-attention factor we can adjust how the model handles
relative positions that are larger than the context length that the model was
trained on. And this is one during inference as opposed to other methods like
LongRoPE which are done during fine-tuning.

Lets try this out with the following input prompt file:
```console
$ ./run-tokenize.sh self-extend.txt
Total number of tokens: 7038
```
And check the module context length which is the length that the model was
trained on:
```console
$ ./inspect-model.sh models/llama-2-7b.Q4_0.gguf 
INFO:gguf-dump:* Loading: models/llama-2-7b.Q4_0.gguf
* File is LITTLE endian, script is running on a LITTLE endian host.
* Dumping 22 key/value pair(s)
      1: UINT32     |        1 | GGUF.version = 2
      2: UINT64     |        1 | GGUF.tensor_count = 291
      3: UINT64     |        1 | GGUF.kv_count = 19
      4: STRING     |        1 | general.architecture = 'llama'
      5: STRING     |        1 | general.name = 'LLaMA v2'
      6: UINT32     |        1 | llama.context_length = 4096
      7: UINT32     |        1 | llama.embedding_length = 4096
      8: UINT32     |        1 | llama.block_count = 32
      9: UINT32     |        1 | llama.feed_forward_length = 11008
     10: UINT32     |        1 | llama.rope.dimension_count = 128
     11: UINT32     |        1 | llama.attention.head_count = 32
     12: UINT32     |        1 | llama.attention.head_count_kv = 32
     13: FLOAT32    |        1 | llama.attention.layer_norm_rms_epsilon = 9.999999747378752e-06
     14: UINT32     |        1 | general.file_type = 2
     15: STRING     |        1 | tokenizer.ggml.model = 'llama'
     16: [STRING]   |    32000 | tokenizer.ggml.tokens
     17: [FLOAT32]  |    32000 | tokenizer.ggml.scores
     18: [INT32]    |    32000 | tokenizer.ggml.token_type
     19: UINT32     |        1 | tokenizer.ggml.bos_token_id = 1
     20: UINT32     |        1 | tokenizer.ggml.eos_token_id = 2
     21: UINT32     |        1 | tokenizer.ggml.unknown_token_id = 0
     22: UINT32     |        1 | general.quantization_version = 2
```
So the model was trained on a context length of `4096` tokens, and we are going
to use an input prompt of size `7038` tokens.

With self-extend we then have to decide what values to set for `grp-attn-n` and
`grp-attn-w`.

```console
./llama-cli -m models/llama-2-7b.Q4_0.gguf -ngl 10 -f self-extend.txt -c 8000 --temp 1 -n 200 --grp-attn-n 4 --grp-attn-w 128
```
So this means that we are going to take each group of 128 tokens and split them
into 4 groups of 32 tokens.
```
   0                          127                       255
   +---------------------------|-------------------------+----------------------+
   [0-31][32-63][64-95][96-127][128-159][160-191][192-223][224-255]
   [grp0][grp1 ][grp2 ][grp3  ][grp4  ][grp5  ][grp6  ][grp7  ]

grp0 = all tokens will have position 0
grp1 = all tokens will have position 1
grp2 = all tokens will have position 2
...
```
In this case will be processing the batch with 2048 tokens and this is also
the context length of the model we are using. So additional tokens would have a
position that is larger than the context length that the model was trained on.
So we want to map these position into the range the model has seen during
training so that the attention mechanism can work as intended.

If we have a grp-attn-n of 2 and grp-attn-w of 2048 each position. We have
window with of 2048:
```
[1720595163] n_past = 2048
[1720595163] embd_inp.size(): 7037, n_consumed: 2048
[1720595163] 
[1720595163] shift: [     0,   2048] +      0 -> [     0,   2048]
[1720595163] div:   [     0,   2048] /      2 -> [     0,   1024]
[1720595163] shift: [  2048,   2048] +  -1024 -> [  1024,   1024]
```
Each position in this range will be mapped using /2. For example:
```
(gdb) p ctx.kv_self.cells[0]
$23 = {pos = 0, delta = 0, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
(gdb) p ctx.kv_self.cells[1]
$24 = {pos = 0, delta = -1, src = 0, seq_id = std::set with 1 element = {[0] = 0}}

(gdb) p ctx.kv_self.cells[2]
$25 = {pos = 1, delta = -1, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
(gdb) p ctx.kv_self.cells[3]
$26 = {pos = 1, delta = -2, src = 0, seq_id = std::set with 1 element = {[0] = 0}}

(gdb) p ctx.kv_self.cells[4]
$27 = {pos = 2, delta = -2, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
(gdb) p ctx.kv_self.cells[5]
$28 = {pos = 2, delta = -3, src = 0, seq_id = std::set with 1 element = {[0] = 0}}

(gdb) p ctx.kv_self.cells[2044]
$32 = {pos = 1022, delta = -1022, src = 0, seq_id = std::set with 1 element = {[0] = 0}}

(gdb) p ctx.kv_self.cells[2045]
$31 = {pos = 1022, delta = -1023, src = 0, seq_id = std::set with 1 element = {[0] = 0}}

(gdb) p ctx.kv_self.cells[2046]
$30 = {pos = 1023, delta = -1023, src = 0, seq_id = std::set with 1 element = {[0] = 0}}

(gdb) p ctx.kv_self.cells[2047]
$29 = {pos = 1023, delta = -1024, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
```
Is we use a grp-attn-n of 4  we would get:
```console
[1720596142] n_past = 2048
[1720596142] embd_inp.size(): 7037, n_consumed: 2048
[1720596142] 
[1720596142] shift: [     0,   2048] +      0 -> [     0,   2048]
[1720596142] div:   [     0,   2048] /      4 -> [     0,    512]
[1720596142] shift: [  2048,   2048] +  -1536 -> [   512,    512]

(gdb) p ctx.kv_self.cells[0]
$2 = {pos = 0, delta = 0, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
(gdb) p ctx.kv_self.cells[1]
$3 = {pos = 0, delta = -1, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
(gdb) p ctx.kv_self.cells[2]
$4 = {pos = 0, delta = -2, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
(gdb) p ctx.kv_self.cells[3]
$5 = {pos = 0, delta = -3, src = 0, seq_id = std::set with 1 element = {[0] = 0}}

(gdb) p ctx.kv_self.cells[4]
$6 = {pos = 1, delta = -3, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
(gdb) p ctx.kv_self.cells[5]
$7 = {pos = 1, delta = -4, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
(gdb) p ctx.kv_self.cells[6]
$8 = {pos = 1, delta = -5, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
(gdb) p ctx.kv_self.cells[7]
$9 = {pos = 1, delta = -6, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
```
So we have control over the number of tokens what will have the same position in
this case two tokens will have the same position.

Can you accomplish the same thing but having a smaller window size?
For example setting the window group length/width to 1024:
```console
[1720601833] n_past = 2048
[1720601833] embd_inp.size(): 7037, n_consumed: 2048
[1720601833] 
[1720601833] shift: [     0,   2048] +      0 -> [     0,   2048]
[1720601833] div:   [     0,   1024] /      2 -> [     0,    512]
[1720601833] shift: [  1024,   2048] +   -512 -> [   512,   1536]


(gdb) p ctx.kv_self.cells[0]
$4 = {pos = 0, delta = 0, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
(gdb) p ctx.kv_self.cells[1]
$5 = {pos = 0, delta = -1, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
(gdb) p ctx.kv_self.cells[2]

$6 = {pos = 1, delta = -1, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
(gdb) p ctx.kv_self.cells[3]
$7 = {pos = 1, delta = -2, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
(gdb) p ctx.kv_self.cells[4]

$8 = {pos = 2, delta = -2, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
(gdb) p ctx.kv_self.cells[5]
$9 = {pos = 2, delta = -3, src = 0, seq_id = std::set with 1 element = {[0] = 0}}

(gdb) p ctx.kv_self.cells[1022]
$15 = {pos = 511, delta = -511, src = 0, seq_id = std::set with 1 element = {[0] = 0}
(gdb) p ctx.kv_self.cells[1023]
$12 = {pos = 511, delta = -512, src = 0, seq_id = std::set with 1 element = {[0] = 0}}

(gdb) p ctx.kv_self.cells[1024]
$13 = {pos = 1024, delta = 0, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
(gdb) p ctx.kv_self.cells[1025]
$14 = {pos = 1025, delta = 0, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
```
Notice that this time only the positions up to 1024 are mapped. But we will
go through the loop once more.
```console
n_past_old = 2048, n_past = 1536, ga_i = 512

[1720602118] 
[1720602119] shift: [   512,   1536] +    512 -> [  1024,   2048]
[1720602120] div:   [  1024,   2048] /      2 -> [   512,   1024]
[1720602120] shift: [  2048,   2048] +  -1024 -> [  1024,   1024]

(gdb) p ctx.kv_self.cells[1024]
$21 = {pos = 512, delta = -512, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
(gdb) p ctx.kv_self.cells[1025]
$22 = {pos = 512, delta = -513, src = 0, seq_id = std::set with 1 element = {[0] = 0}}

(gdb) p ctx.kv_self.cells[1026]
$23 = {pos = 513, delta = -513, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
(gdb) p ctx.kv_self.cells[1027]
$24 = {pos = 513, delta = -514, src = 0, seq_id = std::set with 1 element = {[0] = 0}}

(gdb) p ctx.kv_self.cells[1028]
$25 = {pos = 514, delta = -514, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
(gdb) p ctx.kv_self.cells[1029]
$26 = {pos = 514, delta = -515, src = 0, seq_id = std::set with 1 element = {[0] = 0}
```
So we've accomplished the exact same result with regards to the grouping. I'm
finding it difficult to understand why one would want to have a smaller window
size than the context length that the model was trained on.

One thing to note is that we only enter the self-extend block if `n_past` is
greater or equal to `ga_i + ga_w`:
```c++
                while (n_past >= ga_i + ga_w) {
```

Wait, so with `ga_n`, which is the factor, this will give an x context length
extension.

So the following would enable a context length of 4096 (2x2048):
```
--grp-attn-n 2 --grp-attn-w 2048
```
Notice that for positions up to 4095 we are alright, but above that the
positions will be:
```
4096 / 2 = 2048
4097 / 2 = 2048
4098 / 2 = 2049
4099 / 2 = 2049
```
And these are outside of the context length that this model was trained on.

So to be able to support larger context lengths the `grp-attn-n` and
`grp-attn-w` options can be used. So a `gan` value of 2 and a `gaw` value
of 2048 would mean that we can handle 4096 context lengths (doubling the
context). If we need larger we can increase one or both (depending on the model
used and the context lenght it was trained on) of these values.

In this case the context length the model was trained on was 2048 so that is
the max value we can specify for `grp-attn-w` since we are dividing this with
the number of groups:
```
4096 / 3 = 1365
4097 / 3 = 1365
4098 / 3 = 1366
4099 / 3 = 1366
4100 / 3 = 1366

4101 / 3 = 1367
4102 / 3 = 1367
4103 / 3 = 1367

6144 / 3 = 2048
```
So this would enable context of 3x2048=6144 token. 
And we can increase `gan` to allow longer context.



If we configure self-extend to have many groups, meaning that more positions
are mapped to the same position and the attention mechanism might not be able to
handle this (will mess up the attention scores depending on the model used). 

I think a good default would be to set the width to the context length that
the model was trained on, and the groups to 2 to keep.

The self attention with the `floor` operation is called "grouped attention".

The positions get mapped using the following operation:
```
floor(pos / ga_n)
```
`ga_n` is the number that is used in the floor operation and determines how many
groups the relative positions are divided into. The default is 1 which means
that the relative positions are not divided into groups.

In the following we will have decoded the prompt, which will have populated
the `kv_cache`:
```console
(gdb) p ctx.kv_self.used
$86 = 5
(gdb) p n_past
$167 = 5
```
The first five elements in the cache are:
```console
$94 = {pos = 0, delta = 0, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
$95 = {pos = 1, delta = 0, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
$96 = {pos = 2, delta = 0, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
$97 = {pos = 3, delta = 0, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
$98 = {pos = 4, delta = 0, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
$99 = {pos = -1, delta = 0, src = 0, seq_id = std::set with 0 elements}
```
So we can see that we have 5 cells in the cache and the positions are 0-4. And
I'm showing the fifth cells to show that an empty cell has position -1 and an
empty set of sequence ids.

Lets take a look at some of the variables that will be used shortly:
```console
(gdb) p ga_i
$25 = 0

(gdb) p ga_w
$26 = 4

(gdb) p ga_n
$27 = 2

(gdb) p n_past
$28 = 5
```
So we will continue the following while loop as long as `n_past` is greater
than or equal to `ga_i + ga_w`:
```c++
// context extension via Self-Extend
while (n_past >= ga_i + ga_w) {
    const int ib = (ga_n * ga_i) / ga_w;
    const int bd = (ga_w / ga_n) * (ga_n - 1);
    const int dd = (ga_w / ga_n) - ib*bd - ga_w;

    LOG("\n");
    LOG("shift: [%6d, %6d] + %6d -> [%6d, %6d]\n", ga_i, n_past, ib*bd, ga_i + ib*bd, n_past + ib*bd);
    LOG("div:   [%6d, %6d] / %6d -> [%6d, %6d]\n", ga_i + ib*bd, ga_i + ib*bd + ga_w, ga_n, (ga_i + ib*bd)/ga_n, (ga_i + ib*bd + ga_w)/ga_n);
    LOG("shift: [%6d, %6d] + %6d -> [%6d, %6d]\n", ga_i + ib*bd + ga_w, n_past + ib*bd, dd, ga_i + ib*bd + ga_w + dd, n_past + ib*bd + dd);

    llama_kv_cache_seq_add(ctx, 0, ga_i, n_past, ib*bd);
    llama_kv_cache_seq_div(ctx, 0, ga_i + ib*bd, ga_i + ib*bd + ga_w, ga_n);
    llama_kv_cache_seq_add(ctx, 0, ga_i + ib*bd + ga_w, n_past + ib*bd, dd);

    n_past -= bd;

    ga_i += ga_w/ga_n;

    LOG("\nn_past_old = %d, n_past = %d, ga_i = %d\n\n", n_past + bd, n_past, ga_i);
}
```
So we have specified that `ga_w` is 4 and `ga_n` is 2. To get the number
of groups we divide `ga_w` by `ga_n` which is 2. So we will have 2 groups.

```c+
    const int ib = (ga_n * ga_i) / ga_w;
```
Now, `ga_i` is the index specifying which group we are currently processing.
This will be incremented with the group size which is `ga_w/ga_n`, 2 in this
case. So the first iteration `ga_i` will be 0, 2, 4, 6, 8.
```
ib = (ga_n * ga_i) / ga_w;             // index base
ib = (2    *    0) /    4;             
ib = 0


  [  ga_w_0  ] [  ga_w_1  ][  ga_w_2  ][  ga_w_3  ]
  +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
  +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
  ↑
 [ib] 

ga_i = 0, ib = 0
ga_i = 2, ib = 1
ga_i = 4, ib = 2
ga_i = 6, ib = 3
```
```
bd = (ga_w / ga_n) * (ga_n - 1);
bd = (4    /    2) * (2    - 1);
bd =            2  * 1;
bd = 2

  [  ga_w_0  ] [  ga_w_1  ][  ga_w_2  ][  ga_w_3  ]
  +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
  +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
  -----------> ----------> ----------> ---------->
      bd          bd          bd          bd

ga_i = 0, ib = 0, covers position [0-1]
ga_i = 2, ib = 1, covers position [2-3]
ga_i = 4, ib = 2, covers position [4-5]
ga_i = 6, ib = 3, covers position [6-7]
```

```
dd = (ga_w/ga_n)  - ib*bd   - ga_w;
dd = (group_size) - (block) - total_width;
dd = (ga_w / ga_n) - ib*bd  - ga_w;
dd = (4    /    2) -  0*2   - 4;
dd =             2 -        - 4;
dd = -2
```

Notice that `ga_w / ga_n` gives us the tokens per group. So `ga_w` is the total
number of tokens used for group attention.

Possible suggestions for adding variable to not have to repeat the same
mutliple times:
```c++
int tokens_per_group = ga_w / ga_n;
int group_size = ga_w / ga_n;
```

`ga_n` is directly used in the floor operation to squeeze or map token positions
into a manageable range. For example, if the context length is 512 and
`ga_n` = 4, the floor operation modifies token positions so that sequences longer
than 512 can be dealt with within the confines of this trained range.
`ga_n` is passed into:
```c++
    llama_kv_cache_seq_div(ctx, 0, ga_i + ib*bd, ga_i + ib*bd + ga_w, ga_n);
```

The first thing that happens is that `llama_kv_cache_seq_add` is called:
```c++
llama_kv_cache_seq_add(ctx, 0, ga_i, n_past, ib*bd);
llama_kv_cache_seq_add(ctx, 0, 0, 5, 0);
```
```c++
void llama_kv_cache_seq_add(struct llama_context * ctx,
                            llama_seq_id seq_id,
                            llama_pos p0,
                            llama_pos p1,
                            llama_pos delta) {
    if (delta == 0) {
        return;
    }

    llama_kv_cache_seq_add(ctx->kv_self, seq_id, p0, p1, delta);
}
```
Notice that for this first case delta will be 0 so it will just return as there
is nothing to add. I'll return to this function later and explain it.

Next we have the division (the floor operation):
```c++
                           seq_id,     p0             p1               d
llama_kv_cache_seq_div(ctx, 0    , ga_i + ib*bd, ga_i + ib*bd + ga_w, ga_n);
llama_kv_cache_seq_div(ctx, 0    ,    0 + 0    ,    0 +   0   + 4   ,    2);
```
Now notice how we are using `ga_i` + `ib*bd` 
```
  [  ga_w_0  ] [  ga_w_1  ][  ga_w_2  ][  ga_w_3  ]
  +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
  +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
  ↑
 [p0]
 [ga_i + ib*bd]
 [0    + 0 * 2]
```

```c++
void llama_kv_cache_seq_div(struct llama_context * ctx,
                            llama_seq_id seq_id,
                            llama_pos p0,
                            llama_pos p1,
                            int d) {
    if (d == 1) {
        return;
    }

    llama_kv_cache_seq_div(ctx->kv_self, seq_id, p0, p1, d);
}
```

```console
(gdb) s
llama_kv_cache_seq_div (ctx=0x5555561fa4f0, seq_id=0, p0=0, p1=4, d=2) at src/llama.cpp:18218
```

```c++
static void llama_kv_cache_seq_div(
        struct llama_kv_cache & cache,
                 llama_seq_id   seq_id,
                    llama_pos   p0,
                    llama_pos   p1,
                          int   d) {
    if (p0 < 0) p0 = 0;
    if (p1 < 0) p1 = std::numeric_limits<llama_pos>::max();

    if (cache.recurrent) {
        ...
        return;
    }

    for (uint32_t i = 0; i < cache.size; ++i) {
        if (cache.cells[i].has_seq_id(seq_id) && cache.cells[i].pos >= p0 && cache.cells[i].pos < p1) {
            cache.has_shift = true;

            {
                llama_pos p_old = cache.cells[i].pos;
                cache.cells[i].pos   /= d;
                cache.cells[i].delta += cache.cells[i].pos - p_old;
            }
        }
    }
}
```
The cache size is 8192 in this case we are going to iterate over all of them.
Notice that `cache.cells[i].has_seq_id(seq_id)` checks if a cell has the
passed-in sequence id.

`cache.cells[i].pos >= p0 && cache.cells[i].pos < p1` checks that the cell's
postion is in range, in this case 0->4. In this case the first position cell
will be in the range so we enter the if block and set `has_shift` to true.
```console
(gdb) p p_old
$106 = 0
(gdb) p cache.cells[i].pos
$107 = 0
(gdb) p d
$11 = 2

(gdb) p cache.cells[i].pos /= d
$108 = 0
```
Note that this is /= so we are performing the division and then assigning the
result back to `cache.cells[i].pos`. So this is where we adjust the position and
we are using `ga_n`:
```c++
llama_kv_cache_seq_div(ctx, 0, ga_i + ib*bd, ga_i + ib*bd + ga_w, ga_n);
```
For the next iteration we will have:
```console
$24 = {pos = 0, delta = 0, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
$25 = {pos = 0, delta = -1, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
$26 = {pos = 1, delta = -1, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
$27 = {pos = 1, delta = -2, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
$28 = {pos = 4, delta = 0, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
```

Before and after:
```console
$94 = {pos = 0, delta = 0, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
$95 = {pos = 1, delta = 0, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
$96 = {pos = 2, delta = 0, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
$97 = {pos = 3, delta = 0, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
$98 = {pos = 4, delta = 0, src = 0, seq_id = std::set with 1 element = {[0] = 0}}

$24 = {pos = 0, delta = 0, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
$25 = {pos = 0, delta = -1, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
$26 = {pos = 1, delta = -1, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
$27 = {pos = 1, delta = -2, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
$28 = {pos = 4, delta = 0, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
```
And after we have to go through the rest of the cache because the sequence id
migth be used by other entries.

Back in main we now have:
```c++
llama_kv_cache_seq_add(ctx, 0, ga_i + ib*bd + ga_w, n_past + ib*bd,  dd);
llama_kv_cache_seq_add(ctx, 0,                   4,               5, -2);
```
This time we will actually call through to to:
```console
(gdb) s
llama_kv_cache_seq_add (cache=..., seq_id=0, p0=4, p1=5, delta=-2) at src/llama.cpp:3138
```
```c++
static void llama_kv_cache_seq_add(
        struct llama_kv_cache & cache,
                 llama_seq_id   seq_id,
                    llama_pos   p0,
                    llama_pos   p1,
                    llama_pos   delta) {
    uint32_t new_head = cache.size;

    if (p0 < 0) p0 = 0;
    if (p1 < 0) p1 = std::numeric_limits<llama_pos>::max();

    if (cache.recurrent) {
        ...
        return;
    }

    for (uint32_t i = 0; i < cache.size; ++i) {
        if (cache.cells[i].has_seq_id(seq_id) && cache.cells[i].pos >= p0 && cache.cells[i].pos < p1) {
            cache.has_shift = true;
            cache.cells[i].pos   += delta;
            cache.cells[i].delta += delta;

            if (cache.cells[i].pos < 0) {
                if (!cache.cells[i].is_empty()) {
                    cache.used--;
                }
                cache.cells[i].pos = -1;
                cache.cells[i].seq_id.clear();
                if (new_head == cache.size) {
                    new_head = i;
                }
            }
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    // Otherwise we just start the next search from the beginning.
    cache.head = new_head != cache.size ? new_head : 0;
}
```
Now, just like the div function the for loop and if statement here are the same
and we are going through all the cells in the cache, checking if they have the
seq id and are in the range. But notice that the range is now 4-5 only!
```console
(gdb) p p0
$32 = 4
(gdb) p p1
$33 = 5
```
So the first 4 cells will not be in the range but the 4th will. This will then
set the `has_shift` to true and adjust.
The current position of this cell is 4:
```console
(gdb) p cache.cells[i].pos
$37 = 4
(gdb) p delta
$38 = -2
```
Recall that this is what the cache looked like before entering this function:
```console
$24 = {pos = 0, delta = 0, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
$25 = {pos = 0, delta = -1, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
$26 = {pos = 1, delta = -1, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
$27 = {pos = 1, delta = -2, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
$28 = {pos = 4, delta = 0, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
```
The addition of the delta (which i -2) will change the position from 4 to 2:
```console
(gdb) p cache.cells[i].pos + delta
$39 = 2
```
And then the delta is also updated.
```console
(gdb) p cache.cells[i].delta
$41 = 0
(gdb) p cache.cells[i].delta + delta
$42 = -2
```
Next we have the following but pos is not less than 0 so we will not enter.
```c++
            if (cache.cells[i].pos < 0) {
                if (!cache.cells[i].is_empty()) {
                    cache.used--;
                }
                cache.cells[i].pos = -1;
                cache.cells[i].seq_id.clear();
                if (new_head == cache.size) {
                    new_head = i;
                }
            }
```
Our sequence/range will have completed and the the cells looks like this:
```
$45 = {pos = 0, delta = 0, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
$46 = {pos = 0, delta = -1, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
$47 = {pos = 1, delta = -1, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
$48 = {pos = 1, delta = -2, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
$49 = {pos = 2, delta = -2, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
```
So this last addition adjusted the last position.

Back in the main while look we then have:
```console
                    n_past -= bd;

                    ga_i += ga_w/ga_n;
```
Now, this is interesting `n_past` is currently 5 and `bd` is 2 so `n_past` will
be updated to 3. So instead of having the position of the next token be 5 it
has become 3 (which is the next sequence, following the last pos which was 2).

And `ga_i` will be updated to become 2 (the group size)

So the next time we call `llama_decode` `n_past` will be 3:
```c++
if (llama_decode(ctx, llama_batch_get_one(&embd[i], n_eval, n_past, 0))) {
```
So even though after the prompt was decoded, after which `n_past` was 5 it is
now 3.

After a few decodes `n_past` will again be greater than or equal to `ga_i + ga_w`
and this time the first add will be entered.

This is what the cells look like and `n_past` is 6:
```console
$64 = {pos = 0, delta = 0, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
$65 = {pos = 0, delta = 0, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
$66 = {pos = 1, delta = 0, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
$67 = {pos = 1, delta = 0, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
$68 = {pos = 2, delta = 0, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
$69 = {pos = 3, delta = 0, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
$70 = {pos = 4, delta = 0, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
$71 = {pos = 5, delta = 0, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
$72 = {pos = -1, delta = 0, src = 0, seq_id = std::set with 0 elements}
```
Now, the range is pos 2 -> pos 6 (p0=2, p1=6) so we will be skipping the first
5 cells.
```console
(gdb) p i
$75 = 4
(gdb) p cache.cells[i]
$74 = {pos = 2, delta = 0, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
```
Again we set `has_shift` to true and adjust the position:
```console
$69 = {pos = 3, delta = 0, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
(gdb) p cache.cells[i].pos
$76 = 2
(gdb) p cache.cells[i].pos / delta
$77 = 1
(gdb) p cache.cells[5]
$86 = {pos = 5, delta = 2, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
```
So if a cell has a shift then we can use the delta to get the original position.

So that this shift will have updated the positions in the range 2-6:
```console
From:
$64 = {pos = 0, delta = 0, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
$65 = {pos = 0, delta = 0, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
$66 = {pos = 1, delta = 0, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
$67 = {pos = 1, delta = 0, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
$68 = {pos = 2, delta = 0, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
$69 = {pos = 3, delta = 0, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
$70 = {pos = 4, delta = 0, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
$71 = {pos = 5, delta = 0, src = 0, seq_id = std::set with 1 element = {[0] = 0}}

To:
$88 = {pos = 0, delta = 0, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
$89 = {pos = 0, delta = 0, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
$90 = {pos = 1, delta = 0, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
$91 = {pos = 1, delta = 0, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
  2 = {pos = 4, delta = 2, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
  3 = {pos = 5, delta = 2, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
  4 = {pos = 6, delta = 2, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
  5 = {pos = 7, delta = 2, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
```
Is this done because the positions were added using `n_past` which was 3 and
then incremented, the new cells have the positions 3, 4, 5. This add is
adjusting them to be in the order prior to the adjustment of `n_past`. Notice
that they are now incremental from the first position, But not in the grouping
but that will be handled by the next division operation:
```console
(gdb) s
llama_kv_cache_seq_div (ctx=0x5555561fa4f0, seq_id=0, p0=4, p1=8, d=2) at src/llama.cpp:18218
```
Notice that the position range is now 4-8 which matches the shifted positions
from the addition call.
```console
    for (uint32_t i = 0; i < cache.size; ++i) {
        if (cache.cells[i].has_seq_id(seq_id) && cache.cells[i].pos >= p0 && cache.cells[i].pos < p1) {
            cache.has_shift = true;

            {
                llama_pos p_old = cache.cells[i].pos;
                cache.cells[i].pos   /= d;
                cache.cells[i].delta += cache.cells[i].pos - p_old;
            }
        }
    }
```
When i = 4 we will enter the block and set `has_shift`.
```console
(gdb) p cache.cells[i]
$100 = {pos = 4, delta = 2, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
(gdb) p p_old
$99 = 4

After all adjustments:
(gdb) p cache.cells[4]
$102 = {pos = 2, delta = 0, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
```
After this will will have:
```console
$103 = {pos = 0, delta = 0, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
$104 = {pos = 0, delta = 0, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
$105 = {pos = 1, delta = 0, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
$106 = {pos = 1, delta = 0, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
$107 = {pos = 2, delta = 0, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
$108 = {pos = 2, delta = -1, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
$109 = {pos = 3, delta = -1, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
$110 = {pos = 3, delta = -2, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
```
Back in main we have the last shift:
```console
(gdb) s
llama_kv_cache_seq_add (ctx=0x5555561fa4f0, seq_id=0, p0=8, p1=8, delta=-4) at src/llama.cpp:18210
```
Notice that the range is 8-8 which in our case means that there will be no 
match and no shift!
```console
$117 = {pos = 0, delta = 0, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
$118 = {pos = 0, delta = 0, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
$119 = {pos = 1, delta = 0, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
$120 = {pos = 1, delta = 0, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
$121 = {pos = 2, delta = 0, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
$122 = {pos = 2, delta = -1, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
$123 = {pos = 3, delta = -1, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
$124 = {pos = 3, delta = -2, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
```
And back in main we gain decrement `n_past`:
```c++
(gdb) p n_past
$126 = 6

587	                    n_past -= bd;

(gdb) p n_past
$128 = 4

589	                    ga_i += ga_w/ga_n;

(gdb) p ga_i
$129 = 4
```

So to recap a little...when `llama_decode` is called the first time the tokens
in the batch will be added to the cache (using find slots function). This is
fine as they will be adjusted if needed by the code we've gone through above.

By setting `cache.has_shift` to true when `llama_decode_internal` calls
```c++
        if (hparams.causal_attn) {
            llama_kv_cache_update(&lctx);
            ...
        }
```
```c++
    // apply K-shift if needed
    if (lctx.model.hparams.rope_type != LLAMA_ROPE_TYPE_NONE && lctx.kv_self.has_shift) {
        {
            ggml_backend_sched_reset(lctx.sched);

            ggml_cgraph * gf = llama_build_graph_k_shift(lctx);

            ggml_backend_sched_alloc_graph(lctx.sched, gf);

            llama_set_k_shift(lctx);

            llama_graph_compute(lctx, gf, lctx.cparams.n_threads);

            need_reserve = true;
        }

        {
            auto & kv_self = lctx.kv_self;

            kv_self.has_shift = false;

            for (uint32_t i = 0; i < kv_self.size; ++i) {
                kv_self.cells[i].delta = 0;
            }
        }
    }
```

When the kv-cache `has_shift` is true like in this case where we updated above
in the self-extend code.
```c
static struct ggml_cgraph * llama_build_graph_k_shift(llama_context & lctx) {
    llama_batch dummy;
    dummy.n_tokens = 0;

    llm_build_cb cb = [&](struct ggml_tensor * , const char * , int ) { };

    struct llm_build_context llm(lctx, dummy, cb, false);

    llm.init();

    struct ggml_cgraph * result = llm.build_k_shift();

    llm.free();

    return result;
}
```

```
    struct ggml_cgraph * build_k_shift() {
        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, LLAMA_MAX_NODES, false);

        GGML_ASSERT(kv_self.size == n_ctx);

        lctx.inp_K_shift = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_ctx);
        cb(lctx.inp_K_shift, "K_shift", -1);
        ggml_set_input(lctx.inp_K_shift);

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor * rope_factors = build_rope_factors(il);
            struct ggml_tensor * tmp =
                // we rotate only the first n_rot dimensions
                ggml_rope_ext_inplace(ctx0,
                        ggml_view_3d(ctx0, kv_self.k_l[il],
                            n_embd_head_k, n_head_kv, n_ctx,
                            ggml_row_size(kv_self.k_l[il]->type, n_embd_head_k),
                            ggml_row_size(kv_self.k_l[il]->type, n_embd_k_gqa),
                            0),
                        lctx.inp_K_shift, rope_factors, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow);

            cb(tmp, "K_shifted", il);
            ggml_build_forward_expand(gf, tmp);
        }

        return gf;
    }
```
So first a 1d tensor is created with the same size of the context (8192):
```console
(gdb) p *lctx.inp_K_shift 
$185 = {type = GGML_TYPE_I32,
backend = GGML_BACKEND_TYPE_CPU,
buffer = 0x0,
ne = {8192, 1, 1, 1}, nb = {4, 32768, 32768, 32768},
op = GGML_OP_NONE,
op_params = {0 <repeats 16 times>}, flags = 0, grad = 0x0, src = {
0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0}, view_src = 0x0, view_offs = 0, data = 0x0, 
name = '\000' <repeats 63 times>, extra = 0x0}
```
Next we iterate over all the layers (42), and for each layer we will create
a ggml operation using `ggml_rope_ext_inplace`. This will add an operation for
rope which will update the target tensors which are the `kv_self.k_l[il]`
tensors. And notice that the position tensor, the third argument to
`ggml_rope_ext_inplace` is the `lctx.inp_K_shift` tensor which is the tensor
that was updated above with the delta values from the cache cells. So this is
how the the kv-cache interacts with rope I think, the positions will have been
updated to reflect the changes to the positions, and then a normal rope
operation will follow. For details about rope see [ggml-rope.md](ggml-rope.md).


After that `llama_set_k_shift` is called which sets the _delta_ from the cache
cells on the tensor:
```c

static void llama_set_k_shift(llama_context & lctx) {
    const int64_t kv_size = lctx.kv_self.size;

    assert(ggml_backend_buffer_is_host(lctx.inp_K_shift->buffer));

    int32_t * data = (int32_t *) lctx.inp_K_shift->data;

    for (int i = 0; i < kv_size; ++i) {
        data[i] = lctx.kv_self.cells[i].delta;
    }
}
```
So this is updating the `inp_K_shift` tensor with the delta values from the cache
cells. Later in `llama_decode_internal` we have:
```c
static int llama_decode_internal(
         llama_context & lctx,
           llama_batch   batch_all) { // TODO: rename back to batch
    ...

        // non-causal masks do not use the KV cache
        if (hparams.causal_attn) {
            llama_kv_cache_update(&lctx);

            // if we have enough unused cells before the current head ->
            //   better to start searching from the beginning of the cache, hoping to fill it
            if (kv_self.head > kv_self.used + 2*n_tokens) {
                kv_self.head = 0;
            }

            if (!llama_kv_cache_find_slot(kv_self, u_batch)) {
                return 1;
            }
            ...
        }
```
Notice the call to `llama_kv_cache_update`.

```console
$202 = {pos = 0, delta = 0, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
$203 = {pos = 0, delta = -1, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
$204 = {pos = 1, delta = -1, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
$205 = {pos = 1, delta = -2, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
$206 = {pos = 2, delta = -2, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
```
So I was wrong when I said that it does not look like the function
`build_k_shift` uses the `kv_cache` because it does because it used the tensor
`lctx.inp_K_shift`

```c
    struct ggml_tensor* a = ggml_view_3d(ctx0,
                                         kv_self.k_l[il],
                                         n_embd_head_k,
                                         n_head_kv,
                                         n_ctx,
                                         ggml_row_size(kv_self.k_l[il]->type,
                                         n_embd_head_k),
                                         ggml_row_size(kv_self.k_l[il]->type,
                                         n_embd_k_gqa);
    ggml_rope_ext_inplace(ctx0,
                          a
                          lctx.inp_K_shift,
                          rope_factors,
                          n_rot,
                          rope_type,
                          n_ctx_orig,
                          freq_base,
                          freq_scale,
                          ext_factor,
                          attn_factor,
                          beta_fast,
                          beta_slow);

struct ggml_tensor * ggml_rope_ext_inplace(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b,
        struct ggml_tensor  * c,
        int                   n_dims,
        int                   mode,
        int                   n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow) {
```

So this is rotating the tensor `a` (`kv_self_.k_l[il]`) using the positions in
`b` (`lctx.inp_K_shift`). And recall that the positions in `lctx.inp_K_shift`
were updated using the cache cell deltas.

And after this a normal RoPE operation will be done later in the formard pass
using the positions (which have been adjusted).

Testings this:
So the basic idea is that we have a model which was trained on a certain
context length.
```console
$ ./inspect-model.sh ~/.cache/lm-studio/models/TheBloke/TinyLlama-1.1B-1T-OpenOrca-GGUF/tinyllama-1.1b-1t-openorca.Q2_K.gguf 
INFO:gguf-dump:* Loading: /home/danbev/.cache/lm-studio/models/TheBloke/TinyLlama-1.1B-1T-OpenOrca-GGUF/tinyllama-1.1b-1t-openorca.Q2_K.gguf
* File is LITTLE endian, script is running on a LITTLE endian host.
* Dumping 22 key/value pair(s)
      1: UINT32     |        1 | GGUF.version = 2
      2: UINT64     |        1 | GGUF.tensor_count = 201
      3: UINT64     |        1 | GGUF.kv_count = 19
      4: STRING     |        1 | general.architecture = 'llama'
      5: STRING     |        1 | general.name = 'jeff31415_tinyllama-1.1b-1t-openorca'
      6: UINT32     |        1 | llama.context_length = 2048
      7: UINT32     |        1 | llama.embedding_length = 2048
      8: UINT32     |        1 | llama.block_count = 22
      9: UINT32     |        1 | llama.feed_forward_length = 5632
     10: UINT32     |        1 | llama.rope.dimension_count = 64
     11: UINT32     |        1 | llama.attention.head_count = 32
     12: UINT32     |        1 | llama.attention.head_count_kv = 4
     13: FLOAT32    |        1 | llama.attention.layer_norm_rms_epsilon = 9.999999747378752e-06
     14: UINT32     |        1 | general.file_type = 10
     15: STRING     |        1 | tokenizer.ggml.model = 'llama'
     16: [STRING]   |    32000 | tokenizer.ggml.tokens
     17: [FLOAT32]  |    32000 | tokenizer.ggml.scores
     18: [INT32]    |    32000 | tokenizer.ggml.token_type
     19: UINT32     |        1 | tokenizer.ggml.bos_token_id = 1
     20: UINT32     |        1 | tokenizer.ggml.eos_token_id = 2
     21: UINT32     |        1 | tokenizer.ggml.unknown_token_id = 0
     22: UINT32     |        1 | general.quantization_version = 2
```


### Testing:
I created a text file by downloading a book from the Gutenberg project:
```console
$ wget https://www.gutenberg.org/cache/epub/1184/pg1184.txt
```
I then just kept the first ~600 lines and saved it in a file named
`self-extend.txt`. We can inspect the number of tokens in this files using:
```console
./llama-tokenize -m models/llama-2-7b.Q4_0.gguf -f self-extend.txt --show-count
Total number of tokens: 7038
```
And we can also inspect the models parameters:
```console
$ gguf-py/scripts/gguf-dump.py models/llama-2-7b.Q4_0.gguf 
INFO:gguf-dump:* Loading: models/llama-2-7b.Q4_0.gguf
* File is LITTLE endian, script is running on a LITTLE endian host.
* Dumping 22 key/value pair(s)
      1: UINT32     |        1 | GGUF.version = 2
      2: UINT64     |        1 | GGUF.tensor_count = 291
      3: UINT64     |        1 | GGUF.kv_count = 19
      4: STRING     |        1 | general.architecture = 'llama'
      5: STRING     |        1 | general.name = 'LLaMA v2'
      6: UINT32     |        1 | llama.context_length = 4096
      7: UINT32     |        1 | llama.embedding_length = 4096
      8: UINT32     |        1 | llama.block_count = 32
      9: UINT32     |        1 | llama.feed_forward_length = 11008
     10: UINT32     |        1 | llama.rope.dimension_count = 128
     11: UINT32     |        1 | llama.attention.head_count = 32
     12: UINT32     |        1 | llama.attention.head_count_kv = 32
     13: FLOAT32    |        1 | llama.attention.layer_norm_rms_epsilon = 9.999999747378752e-06
     14: UINT32     |        1 | general.file_type = 2
     15: STRING     |        1 | tokenizer.ggml.model = 'llama'
     ...
```
And notice that this model was trained using a context length of 4096 tokens.

First, lets run this without any self-extend options:
```console
$ ./llama-cli -m models/llama-2-7b.Q4_0.gguf -ngl 10 -f self-extend.txt -c 8192 --temp 0 -n 256
```
This will load the prompt without any issues and initially generation looks
somewhat ok, but then it starts to generate gibberish (just a bunch or new lines).

Next, lets try with the self-extend options:
```console
$ /llama-cli -m models/llama-2-7b.Q4_0.gguf -ngl 10 -f self-extend.txt -c 8192 --temp 0 -n 256 --grp-attn-n 4 --grp-attn-w 256
```
And this worked without any issues and the generation look good.


After the first batch of tokens in the prompt have been decoded the `n_past` is
updated to be the number of tokens in the prompt. In the next iteration the
self-extend code will be called and will update the kv-cache.
In this case `--grp-attn-n 4 --grp-attn-w 256`:
```
[1720155410] n_past = 2048                                                      
[1720155410] embd_inp.size(): 7037, n_consumed: 2048                                
[1720155410]                                                                        
[1720155410] shift: [     0,   2048] +      0 -> [     0,   2048]                  
[1720155410] div:   [     0,    256] /      4 -> [     0,     64]                   
[1720155410] shift: [   256,   2048] +   -192 -> [    64,   1856]               
[1720155410]                                                                    
n_past_old = 2048, n_past = 1856, ga_i = 64
```
The first shift does nothing because the order is correct as this is the first
time decode was run. But the division will divide the positions into groups of
64. The next shift will adjust the positions in the group to be incremental

What confused me was `--grp-attn-n`, the number of groups. Stepping through
the code above where we have a input prompt of 7038 tokens, the first batch
decode will be 2048 tokens, which will set `n_past` to 2048. This will cause
the while loop to start by updating the range 0-64. Since this is the first
time through the first shift does nothing but this is needed later when `n_past`
is updated later in this block. This update will cause the next decode to
incremement the positions using the new self-extend aware cache cells. The
division will take the will take the range of 0-256 and divide those cells
positions by 4, and update their delta with the gt.
We can inspect what the cells look like after the division function returns:
```console
(gdb) p ctx.kv_self.cells[255]
$23 = {pos = 63, delta = -192, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
```
Notice that `pos` is now 63 and the delta is -192 (63 - (-192) = 255)). 
Now, if we take a look at entry 256 we find:
```console
(gdb) p ctx.kv_self.cells[256]
$24 = {pos = 256, delta = 0, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
```
Notice that these positions are not sequential after the division. This is what
the last shift corrects. It will go through the entires cache and update the
sequence id where the ranges is position 256-2048. After this we can again
inspect the cells:
```console
(gdb) p ctx.kv_self.cells[256]
$25 = {pos = 64, delta = -192, src = 0, seq_id = std::set with 1 element = {[0] = 0}}

(gdb) p ctx.kv_self.cells[2047]
$27 = {pos = 1855, delta = -192, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
```
And after this `n_past` is updated to 2048-192 = 1856. This will cause the
while loop to be entered again as it's condition is `n_past >= ga_i + ga_w`
which in this case is `1856 >= 64 + 256` which is true.
This time through the first shift will be performed on the range 64->1856.
```console
(gdb) tbreak  3275
Temporary breakpoint 3 at 0x5555558275c4: file src/llama.cpp, line 3275.
(gdb) continue 
Continuing.

Thread 1 "llama-cli" hit Temporary breakpoint 3, llama_kv_cache_seq_add (cache=..., seq_id=0, p0=64, p1=1856, delta=192) at src/llama.cpp:3275
3275	            cache.has_shift = true;
(gdb) p i
$44 = 256
(gdb) p cache.cells[i]
$45 = {pos = 64, delta = -192, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
```
So we adjusted the remaining positions when we updated during the last iteration
and we could have exited the loop if the conditions were different. But in this
case we need to shift the positions from 64 to the end of the cache. We want
the 'normal/natural' sequence order for these entries, think of this as array
order from 0-2048. So we adjusted the first 64 entries by dividing, and then
adjusted all the values above that as well so this will work with the decode
function. But now we want them in normal order so cell[256] should have pos 256
and so on.
After the update we can inspect the cells:
```console
(gdb) p cache.cells[256]
$50 = {pos = 256, delta = -192, src = 0, seq_id = std::set with 1 element = {[0] = 0}}

(gdb) p cache.cells[257]
$52 = {pos = 65, delta = -192, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
```
Next time though the loop call 257 will be updated to pos 257.
After that the division will operate on the range 256-512.



Configuration parameters:
* prompt token size (to determine if self-extend is required)
* context size (-c) instead of using the context size from the model
* self-extend options

So if we know the size of the initial prompt, we can set the context size to
a value larger than the model was trained on and then use the self-extend
option to generate text.
