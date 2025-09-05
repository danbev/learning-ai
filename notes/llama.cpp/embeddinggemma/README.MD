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
$ tail converted-model-kv-fix-output.txt converted-model-master-output.txt original-model-output.txt
==> converted-model-kv-fix-output.txt <==
embedding 530: -0.945007  0.803382 -0.191294  ...  1.387448 -1.333874  0.637110
embedding 531: -1.757166  0.210851  0.340471  ...  2.249267 -0.892162  0.145052
embedding 532: -0.143790  1.629396 -1.210387  ... -1.926077 -5.181782  1.892621
embedding 533: -0.761957 -0.437432 -0.574391  ...  0.916270 -0.056626 -0.044712
embedding 534: -8.569762  3.575454 -3.283160  ... -5.691900 -3.884746  1.290939

Embeddings size: 410880
Saving logits to data/llamacpp-embeddinggemma-300M-embeddings.bin
Logits saved to data/llamacpp-embeddinggemma-300M-embeddings.bin
Logits saved to data/llamacpp-embeddinggemma-300M-embeddings.txt

==> converted-model-master-output.txt <==
embedding 530: -0.936975  0.735835 -0.166969  ...  1.642935 -1.072979  0.832203
embedding 531: -2.060557  0.003230  0.225303  ...  2.659825 -0.516793  0.710358
embedding 532: -0.048221  1.902493 -1.562916  ... -2.104920 -5.607339  1.913462
embedding 533: -0.875762 -0.494703 -0.560324  ...  0.883945  0.006376  0.086154
embedding 534: -8.583373  3.699894 -4.216038  ... -5.364049 -4.334474  1.325724

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
Lets try setting n_swa to 1024 in our model and the original model.

So the original model should looks like the following (from [original-model-output-1024.txt](original-model-output-1024.txt)):
```console
embedding 340:  2.214964 -0.614099 -0.635387  ... -0.030148 -0.151567  0.021538 
embedding 341:  1.634534 -0.008599 -0.363462  ...  0.439301 -0.032693  0.372896 
embedding 342: -0.639432 -0.373834 -0.564425  ...  0.980423 -0.120268 -0.251030 
embedding 343: -0.656197 -0.133294 -0.325613  ...  0.520328 -0.382065 -0.411596 
embedding 344:  0.640330 -0.313765 -0.263950  ...  0.455049 -0.335423 -0.063616 
embedding 345: -0.107409 -0.167236 -0.479765  ...  0.828804 -0.226719 -0.110937 
embedding 346: -0.629435  0.095102  0.116739  ...  0.321456 -0.228771  0.112080 
embedding 347: -0.134847  0.666895  0.008453  ... -2.546186  0.498469 -1.361205 
embedding 348:  0.087949 -0.068500 -0.291025  ... -0.129099  0.245765 -0.425569 
embedding 349: -0.383739 -0.111206 -0.199367  ...  0.887945 -0.140507  0.145907 
embedding 350: -0.053519 -0.133043 -0.103100  ...  0.328254 -0.180183 -0.033386 
embedding 351: -0.411276  0.187741 -0.063503  ...  0.042519 -0.225569  0.165346 
embedding 352: -0.256238  0.454921 -0.249787  ...  0.309935 -0.408935 -0.124169 
embedding 353: -0.251312 -0.436394 -0.232022  ...  0.804492  0.045657  0.363653 
embedding 354: -2.163234 -0.948659  0.971037  ... -0.778741  0.412524 -1.186021 
embedding 355: -3.311509 -0.366520 -0.380377  ... -1.327883  0.039494 -0.128165 
embedding 356:  1.053928 -0.483520 -1.378280  ...  1.329656  0.639379 -1.259318 
embedding 357: -2.012952 -0.251373  0.046241  ... -0.813937  0.238929 -1.238851 
embedding 358: -0.248367 -0.016308 -0.833480  ...  1.942992  0.907128 -0.521933 
embedding 359:  7.540615 -1.853240  1.010493  ...  8.538274  2.236461 -0.134622 
embedding 360:  0.083363 -0.058258 -0.436443  ...  0.825905  0.153373  0.106944 
```
And the converted model looks like this (from [converted-model-master-output-1024.txt](converted-model-master-output-1024.txt)):
```console
embedding 340:  2.246778 -0.617790 -0.673785  ... -0.083672 -0.167500 -0.009376 
embedding 341:  1.683687  0.020123 -0.376886  ...  0.449750 -0.056429  0.348307 
embedding 342: -0.568065 -0.351469 -0.581153  ...  0.965628 -0.146497 -0.252182 
embedding 343: -0.666167 -0.075676 -0.344300  ...  0.595100 -0.424227 -0.446320 
embedding 344:  0.638782 -0.290845 -0.294437  ...  0.467957 -0.371361 -0.073126 
embedding 345: -0.097467 -0.124055 -0.548983  ...  0.887290 -0.281495 -0.151419 
embedding 346: -0.656509  0.147394  0.093707  ...  0.375859 -0.246319  0.121204 
embedding 347: -0.218146  0.688305  0.001589  ... -2.613571  0.509729 -1.320524 
embedding 348:  0.150503 -0.129391 -0.291536  ... -0.362988  0.237815 -0.555755 
embedding 349: -0.377941 -0.078474 -0.259160  ...  0.945410 -0.176689  0.132735 
embedding 350: -0.047789 -0.131796 -0.141364  ...  0.326248 -0.233070 -0.061239 
embedding 351: -0.450649  0.236492 -0.060060  ...  0.076549 -0.233217  0.181921 
embedding 352: -0.190285  0.434373 -0.275959  ...  0.368673 -0.384268 -0.080436 
embedding 353: -0.225914 -0.422998 -0.253155  ...  0.819276 -0.007636  0.365536 
embedding 354: -2.125682 -0.941128  0.682295  ... -0.327854  0.453828 -1.147384 
embedding 355: -4.042728 -0.881110 -0.202684  ... -1.665129 -0.299086  0.231134 
embedding 356:  1.014242 -0.347617 -1.464311  ...  1.280106  0.581784 -1.410935 
embedding 357: -1.836371 -0.364177  0.017972  ... -0.425835  0.448169 -1.271394 
embedding 358: -0.272829  0.062766 -0.909661  ...  1.988533  0.891771 -0.566438 
embedding 359:  8.238274 -2.066273  1.058972  ...  9.283712  2.344737 -0.101076 
embedding 360:  0.091512 -0.011800 -0.462481  ...  0.782526  0.136517  0.114141 
```
