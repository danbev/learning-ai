## Eagle speculative decoding (Extrapolation Algorithm for Greater Language-model Efficiency)
Framework for Lightweight Autoregressive Decoding.

### Eagle 1
Similar to Medusa, this technique does not require a separate draft model, but
instead adds a small head to the main model.

This is autoregessive, but the standard speculative decoding approach is also
autoregressive so how is this different?  
Standard speculative decoding is just like a smaller model, that predicts the
next token. It works in "token" space so it outputs token ids. Eagle 1 changes
this and instead predicts the next hidden state for the next token.

So compared to [Medusa](./medusa.md) which has separate lm_heads for each
token that is predicted Eagle only has one lm_head. The process looks something
like the following:
```
            main_model: predicts h_t
                           |
            eagle heads(N) +--------------------------------+
                           ↓                                |
                        lm_head(h_t) -> l_t     (logits)    |
                           ↓                                |
                        sample(l_t) -> x_{t+1}  (token id)  |
                           ↓                                |
                        inp_embd = tok_emb(x_{t+1})         |
                           ↓                                |
                        inp_eagle = concat(h_t, inp_embd)   |
                           ↓                                |
                        eagle layer predicts h_{t+1}        |
                           ↓                                |
                           ------------h_{t+1}--------------+
                           |    prediction table:
                           |    x{t+1} : l_t[x_{t+1}]
                           |    x{t+2} : l_t[x_{t+2}]
                           ↓    x{t+3} : l_t[x_{t+3}]
                        batch
                           |tokens[:
                           |  prompt
                           |  x_{t+1}
                           |  x_{t+2}
                           |  x_{t+3}
                           |]
                           ↓
           main_model: processes batch and verify
```
Above is just a simple diagram to try to get an overview of the process. One
major difference is that Eagle 1 uses a static tree structure and often predicts
more than a single token. For example for each iteration of the eagle head we
might do N predictions.

Notice that h_t and inp_embd are actually concatenated together so this will be
a larger vector that the original hidden space.

```
Tree Key:
k = 2
( ) = Hidden State [Vector]
[ ] = Token ID [Sampled]

           (h_t) <-- Main model prediction
             |
      +------+------+
      ↓             ↓
    [cat]         [the]    <-- Top-2 Tokens (x_{t+1})
      |             |
    (h_t+1)       (h_t+1') <-- Eagle Predictions
      |             |
   +--+--+       +--+--+
   ↓     ↓       ↓     ↓
 [sat] [ran]   [end] [sun] <-- Top-2 for x_{t+2}
```
Instead of a single chain we now have an array with this tree. Now if the
prediction is "cat->sat", but the real model wanted "cat-ran" the we would loose
the second prediction. With a tree structure we can keep the "ran" prediction:
```
[cat, dog, sat, ran, end, sun]
```

```
Draft chain: ["The", "cat", "sat", "on"]
Real Models: ["The", "cat", "slept", "..."]
```
Here "sat" is rejected so the rest of the draft model predictions are also
rejected. BUt perhaps the real model would have accepted "on" after "slept" but
with the simple chain we would never get to see that prediction.

Also the tree is not just a simple list, each node contains the cumulative log
probability of the token that was predicted. So it is easy to add to the tree,
we just add the log(probability) of the new token to the cumulative log
probability of the parent.


### Eagle 2
Is very similar to Eagle 1 but the token tree is not static but dynamic and can
be pruned if the cumulative log probability of a branch is too low.

TODO: expand on this.

### Eagle 3
So if we think about the two prior version above we saw that the take the
predicted hidden state h_t and the concatenate it with the token embedding of
the predicted token. In Eagle3 instead of taking just the models output (h_t)
they also take a hidden state from the lower level, a mid level, and a later
level (might be the last but I'm not sure yet).
For example, there is a "extract_layers" parameter that specifies which hidden
states to take:
```console
(venv) spark $ gguf-dump models/EAGLE3-LLaMA3.1-Instruct-8B_fp16.gguf
INFO:gguf-dump:* Loading: models/EAGLE3-LLaMA3.1-Instruct-8B_fp16.gguf
* File is LITTLE endian, script is running on a LITTLE endian host.
* Dumping 37 key/value pair(s)
      1: UINT32     |        1 | GGUF.version = 3
      2: UINT64     |        1 | GGUF.tensor_count = 14
      3: UINT64     |        1 | GGUF.kv_count = 34
      4: STRING     |        1 | general.architecture = 'eagle3'
      5: [INT32]    |        3 | eagle3.extract_layers = [2, 16, 29]
```
And this is set in:
```c++
         case LLM_ARCH_EAGLE3:
             {
                 ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                 // EAGLE3 layer extraction configuration
                 // Use array<int, 4> (has template instantiation), then copy first 3 elements
                 std::array<int, 4> extract_layers_tmp = {};
                 if (!ml.get_key_or_arr(LLM_KV_EAGLE3_EXTRACT_LAYERS, extract_layers_tmp, 3, false)) {
                     throw std::runtime_error("EAGLE3 model requires 'extract_layers' in GGUF metadata");
                 }
                 std::copy_n(extract_layers_tmp.begin(), 3, hparams.eagle3_extract_layers.begin());
                 LLAMA_LOG_INFO("%s: EAGLE3 extract_layers = [%d, %d, %d]\n", __func__,
                                hparams.eagle3_extract_layers[0],
                                hparams.eagle3_extract_layers[1],
                                hparams.eagle3_extract_layers[2]);
```

Now, if they are concatenating all of these together then the input this will be
very large and unpractical. What is done instead is these are merged/fused
together. This is done using a separate encoder in llama.cpp. Then we have the
decoder which is the actual speculator itself, the one that generates draft
tokens.



_wip_
