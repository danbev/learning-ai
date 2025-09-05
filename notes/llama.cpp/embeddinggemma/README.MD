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
