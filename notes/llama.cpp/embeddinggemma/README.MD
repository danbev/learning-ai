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
to see if there is a difference when the number of tokens is less than 512 (356)

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
embedding 351: -0.307814  0.251984 -0.204665  ...  0.462497 -0.357251 -0.015757 
embedding 352: -0.207327  0.763141  0.052609  ...  0.298611 -0.558192 -0.340493 
embedding 353: -0.359861 -0.441878  0.046225  ...  0.995378  0.165687  0.499921 
embedding 354: -1.128285 -0.173786 -0.570532  ...  0.975698 -0.199959 -0.122429 
embedding 355: -3.124373  2.904656 -5.487446  ... -1.106131 -5.366333 -3.785431 

Total values: 273408 (356 tokens × 768 dimensions)

Saved bin embeddings to: data/pytorch-embeddinggemma-300M-embeddings.bin
Saved txt embeddings to: data/pytorch-embeddinggemma-300M-embeddings.txt

==> converted-model-master-output-under-512.txt <==
embedding 351: -0.207553  0.188554 -0.158823  ...  0.369807 -0.218508 -0.139167 
embedding 352: -0.434097  0.543417 -0.111845  ...  0.391635 -0.270442 -0.071264 
embedding 353: -0.328975 -0.360527 -0.061640  ...  0.965943  0.186761  0.414118 
embedding 354: -1.070517 -0.242786 -0.518397  ...  0.945989 -0.057610 -0.047675 
embedding 355: -3.859072  5.318798 -4.914835  ... -1.686410 -4.154724 -3.939769 

Embeddings size: 273408
Saving logits to data/llamacpp-embeddinggemma-300M-embeddings.bin
Logits saved to data/llamacpp-embeddinggemma-300M-embeddings.bin
Logits saved to data/llamacpp-embeddinggemma-300M-embeddings.txt

```
