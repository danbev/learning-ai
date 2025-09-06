### EmbeddingGemma swa issue

### Input larger than 512 token
First run the original model with the [swa-prompt.txt](swq-prompt.txt) input file
to get a token count over 512:
```console
(venv) $ make embedding-run-original-model PROMPTS_FILE=swa-prompt.txt > original-model-output.txt
```

Then produce embedding on the current master branch:
```console
(venv) $ make embedding-run-converted-model PROMPTS_FILE=swa-prompt.txt > converted-model-master-output.txt
```

And then produce the embeddings on the kv-cache-fix-swa branch:
```console
(venv) $ make embedding-run-converted-model PROMPTS_FILE=swa-prompt.txt > converted-model-kv-fix-output.txt
```

```console
(venv) $ tail converted-model-master-output.txt original-model-output.txt
==> converted-model-master-output.txt <==
embedding 530: -0.944624  0.804364 -0.191277  ...  1.387934 -1.334470  0.637015
embedding 531: -1.756230  0.209820  0.338830  ...  2.255193 -0.897826  0.146861
embedding 532: -0.143266  1.629679 -1.210484  ... -1.927280 -5.181623  1.894408
embedding 533: -0.762078 -0.437814 -0.574705  ...  0.916314 -0.056354 -0.044370
embedding 534: -8.561273  3.575560 -3.278111  ... -5.694864 -3.885843  1.288128

Embeddings size: 410880
Saving logits to data/llamacpp-embeddinggemma-300M-embeddings.bin
Logits saved to data/llamacpp-embeddinggemma-300M-embeddings.bin
Logits saved to data/llamacpp-embeddinggemma-300M-embeddings.txt

==> original-model-output.txt <==
embedding 530: -1.365641  0.854506 -0.680635  ...  1.614876 -2.079262  0.673541
embedding 531: -0.652792  0.009626  0.425143  ...  0.120669  0.411146 -0.113556
embedding 532:  0.159455  1.898330 -1.388069  ... -2.024029 -5.794567  1.947388
embedding 533: -0.911882 -0.336220 -0.675046  ...  0.990344  0.070253  0.037242
embedding 534: -6.582545  3.765546 -4.170909  ... -4.495653 -3.551418 -0.431231

Total values: 410880 (535 tokens × 768 dimensions)

Saved bin embeddings to: data/pytorch-embeddinggemma-300M-embeddings.bin
Saved txt embeddings to: data/pytorch-embeddinggemma-300M-embeddings.txt
```
The output files are in the current directory. 

I'm working on adding this flag to be able to verify the logits with a specified
input flag like is used above.


### Input less than 512 token
The following test are using [swa-prompt-under-512.txt)(swa-prompt-under-512.txt)
to see if there is a difference when the number of tokens is less than 512 (249)

Original model:
```console
(venv) $ make embedding-run-original-model PROMPTS_FILE=swa-prompt-under-512.txt.txt > original-model-output-under-512.txt

```

Master:
```console
(venv) $ make embedding-run-converted-model PROMPTS_FILE=swa-prompt-under-512.txt > converted-model-master-output-under-512.txt
```

```console
$ tail original-model-output-under-512.txt converted-model-master-output-under-512.txt
==> original-model-output-under-512.txt <==
embedding 244:  2.863687  0.176305  0.046322  ...  0.493868 -0.927739  1.291054 
embedding 245: -0.852267 -0.424254 -0.266554  ...  0.129614 -1.941182  1.067332 
embedding 246:  1.064751 -1.604648  1.176383  ... -0.564188 -0.539131 -0.577689 
embedding 247: -1.464473  0.062398 -0.680420  ...  1.064425 -0.160460 -0.136243 
embedding 248: -1.870404  2.138104 -3.586237  ...  0.000269 -5.451051 -2.179724 

Total values: 191232 (249 tokens × 768 dimensions)

Saved bin embeddings to: data/pytorch-embeddinggemma-300M-embeddings.bin
Saved txt embeddings to: data/pytorch-embeddinggemma-300M-embeddings.txt

==> converted-model-master-output-under-512.txt <==
embedding 244:  2.859944  0.179061  0.043167  ...  0.500611 -0.925673  1.287761 
embedding 245: -0.852439 -0.423924 -0.268251  ...  0.132087 -1.938597  1.064528 
embedding 246:  1.061418 -1.599676  1.170533  ... -0.560962 -0.538622 -0.573997 
embedding 247: -1.462280  0.060194 -0.679344  ...  1.063302 -0.164439 -0.134073 
embedding 248: -1.869754  2.117375 -3.581502  ... -0.004962 -5.455530 -2.189540 

Embeddings size: 191232
Saving logits to data/llamacpp-embeddinggemma-300M-embeddings.bin
Logits saved to data/llamacpp-embeddinggemma-300M-embeddings.bin
Logits saved to data/llamacpp-embeddinggemma-300M-embeddings.txt
```

So if we have a prompt that stays under `n_swa/2`, so 512/2=256 in our case
the logits are very close. But for logits larger than 256, there is a difference.
So when we have an input prompt that is under 256, for example the one I'm using
for testing is 248 token. And the similarity is great. This because it all the
tokens are within the sliding window and there is really no sliding going on?
So every token can attend to every other token, so this is like bi-directional
attention.


### SWA mask
I've tried to inspect the swa mask by printing it out. First I set the `n_swa`
to 10 to be able to see the mask:
```c++
        case LLM_ARCH_GEMMA_EMBEDDING:
            {
                hparams.swa_type = LLAMA_SWA_TYPE_SYMMETRIC;
                hparams.set_swa_pattern(6);

                hparams.causal_attn = false; // embeddings do not use causal attention
                hparams.rope_freq_base_train_swa  = 10000.0f;
                hparams.rope_freq_scale_train_swa = 1.0f;

                ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,    hparams.n_swa);
                hparams.n_swa = 10;
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_POOLING_TYPE,                hparams.pooling_type);

                switch (hparams.n_layer) {
                    case 24: type = LLM_TYPE_0_3B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
                hparams.f_attention_scale = 1.0f / std::sqrt(float(hparams.n_embd_head_k));

            } break;
```

```console
print_mask: === Attention mask ===
print_mask: n_swa : 10, n_kv: 256, swq_type: LLAMA_SWA_TYPE_SYMMETRIC
print_mask: '0' = can attend, '∞' = masked
print_mask: Rows = query tokens, Columns = key/value tokens

  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19
  0  0 0 0 0 0 0 ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞
  1  0 0 0 0 0 0 0 ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞
  2  0 0 0 0 0 0 0 0 ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞
  3  0 0 0 0 0 0 0 0 0 ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞
  4  0 0 0 0 0 0 0 0 0 0 ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞
  5  0 0 0 0 0 0 0 0 0 0 0 ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞
  6  ∞ 0 0 0 0 0 0 0 0 0 0 0 ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞
  7  ∞ ∞ 0 0 0 0 0 0 0 0 0 0 0 ∞ ∞ ∞ ∞ ∞ ∞ ∞
  8  ∞ ∞ ∞ 0 0 0 0 0 0 0 0 0 0 0 ∞ ∞ ∞ ∞ ∞ ∞
  9  ∞ ∞ ∞ ∞ 0 0 0 0 0 0 0 0 0 0 0 ∞ ∞ ∞ ∞ ∞
 10  ∞ ∞ ∞ ∞ ∞ 0 0 0 0 0 0 0 0 0 0 0 ∞ ∞ ∞ ∞
 11  ∞ ∞ ∞ ∞ ∞ ∞ 0 0 0 0 0 0 0 0 0 0 0 ∞ ∞ ∞
 12  ∞ ∞ ∞ ∞ ∞ ∞ ∞ 0 0 0 0 0 0 0 0 0 0 0 ∞ ∞
 13  ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ 0 0 0 0 0 0 0 0 0 0 0 ∞
 14  ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ 0 0 0 0 0 0 0 0 0 0 0
 15  ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ 0 0 0 0 0 0 0 0 0 0
 16  ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ 0 0 0 0 0 0 0 0 0
 17  ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ 0 0 0 0 0 0 0 0
 18  ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ 0 0 0 0 0 0 0
 19  ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ 0 0 0 0 0 0
```
This will of course produce incorrect logits as tokens can now only attend to
a +- 5 token window instead of +- 256, but the mask itself looks alright.

### swa_layers
The swa_layers are set like this:
```c++
    hparams.swa_type = LLAMA_SWA_TYPE_SYMMETRIC;
    hparams.set_swa_pattern(6);
```
And the result is:
```console
(gdb) p hparams.swa_layers 
$3 = {
  _M_elems = {true, true, true, true, true, false,
	            True, true, true, true, true, false,
							true, true, true, true, true, false,
							true, true, true, true, true, false <repeats 489 times>}

(gdb) p hparams.swa_type
$4 = LLAMA_SWA_TYPE_SYMMETRIC
```
And the original model has:
```
  "layer_types": [
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "full_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "full_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "full_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "full_attention"
  ],
```

### rope
train:
```console
(gdb) p hparams.rope_freq_base_train
$5 = 1000000

(gdb) p hparams.rope_freq_scale_train
$6 = 1
```
swa:
```console
(gdb) p hparams.rope_freq_base_train_swa
$7 = 10000

(gdb) p hparams.rope_freq_scale_train_swa 
$9 = 1
```

### setting n_swa = 1024 in both model
Lets try setting n_swa to 1024 in our model and the original model and using a
file that generates 487 tokens ([swa-prompt-450ish.txt](swa-prompt-450ish.txt).

So the original model should looks like the following (from [original-model-output-1024.txt](original-model-output-1024.txt)):
```console
embedding 340:  2.263364 -0.626238 -0.681919  ... -0.070354 -0.165269 -0.016660 
embedding 341:  1.694804  0.009964 -0.375778  ...  0.478708 -0.022504  0.362146 
embedding 342: -0.796452 -0.284255 -0.471018  ...  0.729284 -0.281460 -0.316759 
embedding 343: -0.661255 -0.064119 -0.242731  ...  0.642183 -0.425104 -0.319293 
embedding 344:  0.618559 -0.297795 -0.290502  ...  0.494889 -0.370186 -0.069520 
embedding 345: -0.119944 -0.122810 -0.528405  ...  0.913667 -0.287425 -0.166592 
embedding 346: -0.671322  0.122216  0.089923  ...  0.416883 -0.251089  0.146754 
embedding 347: -0.129827  0.731279 -0.030848  ... -2.718428  0.536977 -1.393983 
embedding 348:  0.460732 -0.643308 -0.174827  ... -1.325012  0.573713 -0.794780 
embedding 349: -0.388479 -0.071489 -0.232057  ...  1.018521 -0.192548  0.146107 
embedding 350: -0.053682 -0.148590 -0.132960  ...  0.365490 -0.235571 -0.067334 
embedding 351: -0.481789  0.232704 -0.053784  ...  0.132114 -0.238493  0.203329 
embedding 352: -0.189264  0.440582 -0.220007  ...  0.384455 -0.410595 -0.072886 
embedding 353: -0.269381 -0.414147 -0.243759  ...  0.815850  0.007568  0.351561 
embedding 354: -2.195405 -0.624633  0.427069  ... -0.278322  0.377566 -0.944792 
embedding 355: -4.233032 -0.828760 -0.452799  ... -1.412385 -0.707759  0.221647 
embedding 356:  0.973853 -0.471105 -1.862583  ...  1.627794  0.684365 -1.387210 
embedding 357: -2.089341 -0.228542 -0.002363  ... -0.435703  0.318262 -1.031783 
embedding 358: -0.418160 -0.040921 -0.966870  ...  2.075976  0.902598 -0.679716 
embedding 359:  8.377390 -2.176193  1.226825  ...  9.309125  2.102131 -0.107556 
embedding 360:  0.120475 -0.052871 -0.465790  ...  0.851682  0.096430  0.068262 
```

And the converted model looks like this (from [converted-model-master-output-1024.txt](converted-model-master-output-1024.txt)):
```console
embedding 340:  2.259687 -0.627697 -0.682402  ... -0.074217 -0.168711 -0.016863 
embedding 341:  1.700005  0.008513 -0.375293  ...  0.478596 -0.021967  0.362356 
embedding 342: -0.815718 -0.235289 -0.448989  ...  0.659585 -0.297617 -0.320333 
embedding 343: -0.667417 -0.066961 -0.229542  ...  0.636291 -0.418103 -0.293302 
embedding 344:  0.620240 -0.301105 -0.290587  ...  0.493859 -0.370333 -0.069547 
embedding 345: -0.118826 -0.125382 -0.528076  ...  0.915655 -0.288920 -0.165299 
embedding 346: -0.668726  0.122790  0.089827  ...  0.423573 -0.249380  0.146841 
embedding 347: -0.136561  0.726108 -0.035495  ... -2.708240  0.538998 -1.388334 
embedding 348:  0.489671 -0.671873 -0.166149  ... -1.390330  0.602278 -0.806077 
embedding 349: -0.388675 -0.070886 -0.231027  ...  1.021425 -0.192995  0.147460 
embedding 350: -0.052111 -0.149906 -0.133442  ...  0.363484 -0.238363 -0.068185 
embedding 351: -0.480296  0.234009 -0.056558  ...  0.138042 -0.238873  0.202796 
embedding 352: -0.188563  0.446667 -0.218543  ...  0.383494 -0.413583 -0.079846 
embedding 353: -0.270329 -0.416484 -0.242730  ...  0.818428  0.009368  0.353297 
embedding 354: -2.196751 -0.601363  0.423901  ... -0.285881  0.379800 -0.951097 
embedding 355: -4.252769 -0.822926 -0.466815  ... -1.433794 -0.712097  0.252976 
embedding 356:  0.968994 -0.467981 -1.863734  ...  1.617094  0.683829 -1.375044 
embedding 357: -2.091579 -0.214106  0.004028  ... -0.444862  0.310037 -1.028754 
embedding 358: -0.413006 -0.043046 -0.966153  ...  2.073784  0.892381 -0.669618 
embedding 359:  8.359043 -2.163706  1.226942  ...  9.276937  2.098100 -0.105900 
embedding 360:  0.126373 -0.057128 -0.466475  ...  0.847704  0.096397  0.068322 
```

So these look much more similar and there is now sliding window attention in
effect here. But when we have sliding window attention we do have an issue.


So perhaps we can simplify this a bit. Lets set the sliding window to say 8
and pass in a small sequence, say 

```console
(venv) $ make embedding-run-converted-model PROMPTS_FILE=small.txt
...

Input prompt: "Hello world"
Tokenized prompt (4 tokens): <bos>Hello world<eos>
output_reserve: reallocating output buffer from size 1.00 MiB to 4.01 MiB
print_mask: === Attention mask ===
print_mask: n_swa : 8, n_kv: 256, swq_type: LLAMA_SWA_TYPE_SYMMETRIC
print_mask: '0' = can attend, '∞' = masked
print_mask: Rows = query tokens, Columns = key/value tokens

      0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19
  0  0 0 0 0 ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞
  1  0 0 0 0 ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞
  2  0 0 0 0 ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞
  3  0 0 0 0 ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞
print_mask: === Attention mask ===
print_mask: n_swa : 8, n_kv: 256, swq_type: LLAMA_SWA_TYPE_SYMMETRIC
print_mask: '0' = can attend, '∞' = masked
print_mask: Rows = query tokens, Columns = key/value tokens

      0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19
  0  0 0 0 0 ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞
  1  0 0 0 0 ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞
  2  0 0 0 0 ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞
  3  0 0 0 0 ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞
Embedding dimension: 768

embedding 0:  5.455087  0.000816 -4.336220  ... -0.844402  1.848774  2.493284
embedding 1:  1.845338 -1.204715 -1.719462  ...  3.425535 -0.171521 -3.412422
embedding 2:  2.196092  0.070246  2.673058  ...  3.277176 -3.908677 -1.083458
embedding 3:  1.150332  5.396164 -2.062721  ... -0.585352 -2.658022 -0.727415

Embeddings size: 3072
Saving logits to data/llamacpp-embeddinggemma-300M-embeddings.bin
Logits saved to data/llamacpp-embeddinggemma-300M-embeddings.bin
Logits saved to data/llamacpp-embeddinggemma-300M-embeddings.txt

```
```console
(venv) $ make embedding-run-original-model PROMPTS_FILE=small.txt
Modified sliding window: 512 -> 8
Importing unreleased model module: transformers.models.tinygemma.modular_tinygemma
Model class: <class 'transformers.models.tinygemma.modular_tinygemma.TinyGemmaModel'>
Model's sliding_window: 8
     2 -> '<bos>'
  9259 -> 'Hello'
  1902 -> '▁world'
     1 -> '<eos>'
Hidden states shape: torch.Size([1, 4, 768])
All embeddings shape: (4, 768)
Embedding dimension: 768

embedding 0:  5.452815  0.000916 -4.332937  ... -0.841584  1.848118  2.494220
embedding 1:  1.847258 -1.203779 -1.719193  ...  3.424226 -0.171834 -3.415397
embedding 2:  2.198046  0.070656  2.672585  ...  3.270103 -3.907986 -1.084595
embedding 3:  1.151672  5.395107 -2.060828  ... -0.583885 -2.657149 -0.725247

Total values: 3072 (4 tokens × 768 dimensions)

Saved bin embeddings to: data/pytorch-embeddinggemma-300M-embeddings.bin
Saved txt embeddings to: data/pytorch-embeddinggemma-300M-embeddings.txt
```
So lets try a file is need to attend to more that just the sliding window:
```console
Input prompt: "Hello world something else"
Tokenized prompt (6 tokens): <bos>Hello world something else<eos>
output_reserve: reallocating output buffer from size 1.00 MiB to 6.02 MiB
print_mask: === Attention mask ===
print_mask: n_swa : 8, n_kv: 256, swq_type: LLAMA_SWA_TYPE_SYMMETRIC
print_mask: '0' = can attend, '∞' = masked
print_mask: Rows = query tokens, Columns = key/value tokens

      0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19
  0  0 0 0 0 0 0 ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞
  1  0 0 0 0 0 0 ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞
  2  0 0 0 0 0 0 ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞
  3  0 0 0 0 0 0 ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞
  4  0 0 0 0 0 0 ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞
  5  0 0 0 0 0 0 ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞
print_mask: === Attention mask ===
print_mask: n_swa : 8, n_kv: 256, swq_type: LLAMA_SWA_TYPE_SYMMETRIC
print_mask: '0' = can attend, '∞' = masked
print_mask: Rows = query tokens, Columns = key/value tokens

     0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
  0  0 0 0 0 0 ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞
  1  0 0 0 0 0 0 ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞
  2  0 0 0 0 0 0 ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞
  3  0 0 0 0 0 0 ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞
  4  0 0 0 0 0 0 ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞
  5  ∞ 0 0 0 0 0 ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞
Embedding dimension: 768

embedding 0:  2.602333 -2.891751 -2.315634  ...  1.318493  0.897715  0.594985 
embedding 1:  3.037957 -1.379041 -1.898681  ...  1.015764  0.901451 -2.756655 
embedding 2:  1.966860  0.872657  2.299456  ...  0.977531 -3.802025 -1.013076 
embedding 3:  0.623389 -0.068921  0.715790  ... -1.001790  0.278366  0.092580 
embedding 4: -4.308455 -6.309823  0.564038  ...  6.871315 -1.384741  2.358924 
embedding 5:  0.319055  3.211755 -1.416538  ... -0.784961 -2.840520 -1.978701
```
So here we have 6 tokens in our sequence, and our sliding window is 8, and half
of that is 4. And this looks correct, token 0 can attend to 4 tokens (in front
of it), token 1 can attend to one token behind and 4 tokens ahead and so on.

And we can run the same using the original model so see the output of this:
```console
(venv) $ make embedding-run-original-model PROMPTS_FILE=small.txt 
Modified sliding window: 512 -> 8
Importing unreleased model module: transformers.models.tinygemma.modular_tinygemma
Model class: <class 'transformers.models.tinygemma.modular_tinygemma.TinyGemmaModel'>
Model's sliding_window: 8
     2 -> '<bos>'
  9259 -> 'Hello'
  1902 -> '▁world'
  2613 -> '▁something'
  1663 -> '▁else'
     1 -> '<eos>'
Hidden states shape: torch.Size([1, 6, 768])
All embeddings shape: (6, 768)
Embedding dimension: 768

embedding 0:  2.710024 -2.705342 -2.352060  ...  1.065063  0.977578  0.710636 
embedding 1:  2.956601 -1.359099 -1.932275  ...  0.539208  0.852095 -2.668386 
embedding 2:  2.137398  0.987929  2.309458  ...  0.849524 -3.789407 -1.269739 
embedding 3:  0.647859 -0.163835  0.550380  ... -0.888664  0.209898  0.132741 
embedding 4: -4.077771 -6.125317  0.509602  ...  6.976178 -1.242871  2.407804 
embedding 5:  0.085943  3.774835 -1.416385  ... -1.497456 -2.533030 -1.415170 

Total values: 4608 (6 tokens × 768 dimensions)
```
Now, this differ quit bit similar to what we say before.

Hmm, one thing that stuck me is that I've been using the transformer from
huggingface and perhaps it has an old implementation.
```console
$ pip install git+https://github.com/huggingface/transformers@v4.56.0-Embedding-Gemma-preview
```
And trying to run that will produce this error:
```console
(venv) $ make embedding-run-original-model PROMPTS_FILE=small.txt 
Modified sliding window: 512 -> 8
Model class: <class 'transformers.models.gemma3.modeling_gemma3.Gemma3TextModel'>
Model's sliding_window: 8
     2 -> '<bos>'
  9259 -> 'Hello'
  1902 -> '▁world'
  2613 -> '▁something'
  1663 -> '▁else'
     1 -> '<eos>'
Traceback (most recent call last):
  File "/home/danbev/work/ai/llama.cpp/examples/model-conversion/./scripts/embedding/run-original-model.py", line 90, in <module>
    outputs = model(**encoded)
              ^^^^^^^^^^^^^^^^
  File "/home/danbev/work/ai/llama.cpp/venv/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/danbev/work/ai/llama.cpp/venv/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1562, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/danbev/work/ai/llama.cpp/venv/lib/python3.11/site-packages/transformers/utils/generic.py", line 1079, in wrapper
    outputs = func(self, *args, **kwargs)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/danbev/work/ai/llama.cpp/venv/lib/python3.11/site-packages/transformers/models/gemma3/modeling_gemma3.py", line 555, in forward
    "full_attention": create_causal_mask(**mask_kwargs),
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/danbev/work/ai/llama.cpp/venv/lib/python3.11/site-packages/transformers/masking_utils.py", line 812, in create_causal_mask
    raise ValueError("Using `or_mask_function` or `and_mask_function` arguments require torch>=2.6")
ValueError: Using `or_mask_function` or `and_mask_function` arguments require torch>=2.6
make: *** [Makefile:121: embedding-run-original-model] Error 1
```
So let upgrate pytorch:
```console
$ pip install torch==2.6.0
```

And running with this version the output is:
```console
(venv) $ make embedding-run-original-model PROMPTS_FILE=small.txt
Modified sliding window: 512 -> 8
Model class: <class 'transformers.models.gemma3.modeling_gemma3.Gemma3TextModel'>
Model's sliding_window: 8
     2 -> '<bos>'
  9259 -> 'Hello'
  1902 -> '▁world'
  2613 -> '▁something'
  1663 -> '▁else'
     1 -> '<eos>'
Hidden states shape: torch.Size([1, 6, 768])
All embeddings shape: (6, 768)
Embedding dimension: 768

embedding 0:  2.710024 -2.705342 -2.352060  ...  1.065063  0.977578  0.710636
embedding 1:  2.956601 -1.359099 -1.932275  ...  0.539208  0.852095 -2.668386
embedding 2:  2.137398  0.987929  2.309458  ...  0.849524 -3.789407 -1.269739
embedding 3:  0.647859 -0.163835  0.550380  ... -0.888664  0.209898  0.132741
embedding 4: -4.077771 -6.125317  0.509602  ...  6.976178 -1.242871  2.407804
embedding 5:  0.085943  3.774835 -1.416385  ... -1.497456 -2.533030 -1.415170

Total values: 4608 (6 tokens × 768 dimensions)

Saved bin embeddings to: data/pytorch-embeddinggemma-300M-embeddings.bin
Saved txt embeddings to: data/pytorch-embeddinggemma-300M-embeddings.txt
```
And to compare with the output from our converted model:
```
embedding 0:  2.710024 -2.705342 -2.352060  ...  1.065063  0.977578  0.710636 
embedding 1:  2.956601 -1.359099 -1.932275  ...  0.539208  0.852095 -2.668386 
embedding 2:  2.137398  0.987929  2.309458  ...  0.849524 -3.789407 -1.269739 
embedding 3:  0.647859 -0.163835  0.550380  ... -0.888664  0.209898  0.132741 
embedding 4: -4.077771 -6.125317  0.509602  ...  6.976178 -1.242871  2.407804 
embedding 5:  0.085943  3.774835 -1.416385  ... -1.497456 -2.533030 -1.415170 
```

## transformers version
During development I has using a copy of the transformers python package to get
started quickly as there was an issue with getting access to an internal git
repository. The was a zipped development version of the transformers library.

Now, I made a mistake and forgot that I had this installed. I had looked at
the requirements file that the convert_hf_to_gguf.py file uses and it looks like
this:
```
mistral-common>=1.8.3

-r ./requirements-convert_legacy_llama.txt
--extra-index-url https://download.pytorch.org/whl/cpu
torch~=2.6.0; platform_machine != "s390x"

# torch s390x packages can only be found from nightly builds
--extra-index-url https://download.pytorch.org/whl/nightly
torch>=0.0.0.dev0; platform_machine == "s390x"
```
And requirements-convert_legacy_llama.txt contains:
```
numpy~=1.26.4
sentencepiece~=0.2.0
transformers>=4.45.1,<5.0.0
gguf>=0.1.0
protobuf>=4.21.0,<5.0.0
```
So my thinking was that nothing would be needed to be updated as anyone
installing these would get a new version of the transformers library.
This was released in:  
https://github.com/huggingface/transformers/releases/tag/v4.56.0-Embedding-Gemma-preview

Interestingly if we look at the last commit we find:  
https://github.com/huggingface/transformers/commit/60b68e304cf4b6569b0660a13b558b929d4b0e77

There was a swa fix which might be related to this issue and show be looked
into:  
https://github.com/huggingface/transformers/pull/40700

So because I've been using the older development version, and also the possibility
that there was an issue with swa in the transformers library (the preview that
I was using (the above was commited 12 hours ago as of this writing) this might
explain why I was seen these strange results locally.

I created a new virtual environment on my machine just now using the following
command:
```console
$ rm -rf venv
$ python3.11 -m venv venv
$ source venv/bin/activate
(venv) $ pip install -r requirements/requirements-convert_hf_to_gguf.txt 
(venv) pip list
Package                   Version
------------------------- ---------
annotated-types           0.7.0
attrs                     25.3.0
certifi                   2025.8.3
charset-normalizer        3.4.3
filelock                  3.19.1
fsspec                    2025.9.0
gguf                      0.17.1
hf-xet                    1.1.9
huggingface-hub           0.34.4
idna                      3.10
Jinja2                    3.1.6
jsonschema                4.25.1
jsonschema-specifications 2025.4.1
MarkupSafe                3.0.2
mistral_common            1.8.4
mpmath                    1.3.0
networkx                  3.5
numpy                     1.26.4
packaging                 25.0
pillow                    11.3.0
pip                       24.0
protobuf                  4.25.8
pycountry                 24.6.1
pydantic                  2.11.7
pydantic_core             2.33.2
pydantic-extra-types      2.10.5
PyYAML                    6.0.2
referencing               0.36.2
regex                     2025.9.1
requests                  2.32.5
rpds-py                   0.27.1
safetensors               0.6.2
sentencepiece             0.2.1
setuptools                65.5.0
sympy                     1.14.0
tiktoken                  0.11.0
tokenizers                0.22.0
torch                     2.4.1+cpu
tqdm                      4.67.1
transformers              4.56.1
typing_extensions         4.15.0
typing-inspection         0.4.1
urllib3                   2.5.0
```
So this will install transformers `4.56.1`, but EmbeddingGemma is a prerelease
and has to be installed using:
```console
(venv) pip install git+https://github.com/huggingface/transformers@v4.56.0-Embedding-Gemma-preview
(venv) $ pip show transformers
Name: transformers
Version: 4.57.0.dev0
Summary: State-of-the-art Machine Learning for JAX, PyTorch and TensorFlow
Home-page: https://github.com/huggingface/transformers
Author: The Hugging Face team (past and future) with the help of all our contributors (https://github.com/huggingface/transformers/graphs/contributors)
Author-email: transformers@huggingface.co
License: Apache 2.0 License
Location: /home/danbev/work/ai/llama.cpp/venv/lib/python3.11/site-packages
Requires: filelock, huggingface-hub, numpy, packaging, pyyaml, regex, requests, safetensors, tokenizers, tqdm
Required-by:
```
