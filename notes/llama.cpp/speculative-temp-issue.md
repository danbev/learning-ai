### Speculate decoding issue
Issue: https://github.com/ggml-org/llama.cpp/issues/29975

So the example provides is a small model with lang vocab of only 8:
```console
     15: UINT32     |        1 | llama.vocab_size = 8
 ```

### Reproduction
```console
$ cat run-repro.sh
build_dir=build-metal-debug
cmd=llama-speculative
python3 repro_l3.py $build_dir/bin/$cmd models/target.gguf models/draft.gguf 1000
```

So this perform 1000 runs and have different seed each time.
 ```console
 $ ./run-repro.sh
seeds 1..1000; runs with first token 'f' (5): 819
  next token 4 ('e'): 1.000   target: 0.345
  next token 6 ('g'): 0.000   target: 0.452
  next token 7 ('h'): 0.000   target: 0.203
```
Notice here that the next token is always 4 ('e') instead of 34.5% of the time
which is what the target model predicts. Keep in mind that the draft model is
only there to guess tokens and is used for speed. The accept/reject part is
designed so that no matter how good or bad the draft model is the final token
that comes out is statistically indistinguishable from what the target model
would have produced on its own. So a bad draft model will have a negative impact
on speed but not on the quality of the final output.

So for this case if we run the target model alone 1000 times we would expect
roughly 345 e tokens, 452 g tokens and 203 h tokens. And like wise if we run the
target+draft via llama-speculative 1000 times we would expect roughly the same
distribution of tokens.

The prompt uses is `bcdbc` and the number of tokens to draft is 4. So the target
model will process the prompt and generatet 1 token, and then the draft model
will draft 4 tokens. This is why we see the following output when running:
```console
bcdbcfegdh
```

Using a seed of 1000 and enabling logging we can see:
```console
0.00.111.655 D all drafted tokens were rejected, sampling from residual distribution
f
0.00.111.662 D the sampled target token (5, 'f') did not match, or we ran out of drafted tokens
0.00.111.662 D keeping sequence 0, n_past_tgt = 5, n_past_dft = 5
0.00.112.917 D  - draft candidate   0 for seq   0, pos   0:      4 (   0.669) 'e'
0.00.112.918 D  - draft candidate   1 for seq   0, pos   0:      6 (   0.238) 'g'
0.00.112.918 D  - draft candidate   2 for seq   0, pos   0:      0 (   0.093) 'a'
0.00.113.691 D  - draft candidate   0 for seq   0, pos   1:      6 (   0.598) 'g'
0.00.113.693 D  - draft candidate   1 for seq   0, pos   1:      2 (   0.227) 'c'
0.00.113.693 D  - draft candidate   2 for seq   0, pos   1:      4 (   0.175) 'e'
0.00.114.443 D  - draft candidate   0 for seq   0, pos   2:      3 (   0.945) 'd'
0.00.114.444 D  - draft candidate   1 for seq   0, pos   2:      6 (   0.030) 'g'
0.00.114.445 D  - draft candidate   2 for seq   0, pos   2:      1 (   0.025) 'b'

````

```console
(lldb) p dist_tgt.data[0]
(llama_token_data)  (id = 5, logit = 3.35087276, p = 0.82117784)
(lldb) p dist_tgt.data[1]
(llama_token_data)  (id = 3, logit = 1.56315172, p = 0.137416795)
(lldb) p dist_tgt.data[2]
(llama_token_data)  (id = 2, logit = 0.363543481, p = 0.0414053649)
```

```console
(lldb) p cur_p->data[0]
(llama_token_data)  (id = 4, logit = 3.38620973, p = 0.668726742)
(lldb) p cur_p->data[1]
(llama_token_data)  (id = 6, logit = 2.3550868, p = 0.238472164)
(lldb) p cur_p->data[2]
(llama_token_data)  (id = 0, logit = 1.41129279, p = 0.092801109)
```
