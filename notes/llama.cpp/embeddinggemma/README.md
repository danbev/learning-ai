### EmbeddingGemma swa issue
This page documents an issue we found in EmbeddingGemma, or at least an issue
I was having locally when verifying the converted model.

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
But Embedding Gemma was released as a preview release:  
https://github.com/huggingface/transformers/releases/tag/v4.56.0-Embedding-Gemma-preview

So anyone just installing the requirements file would not get this model but
would be able to convertet without errors but the model would not be correct.

Interestingly if we look at the last commit we find:  
https://github.com/huggingface/transformers/commit/60b68e304cf4b6569b0660a13b558b929d4b0e77

There was a swa fix which might be related to this issue and show be looked into:  
https://github.com/huggingface/transformers/pull/40700

So because I've been using the older development version, and also the possibility
that there was an issue with swa in the transformers library (the preview that
I was using (the above was commited 12 hours ago as of this writing) this might
have contributed to some of the results we were seeing previously.

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

I'll open a pull request with these changes to the requirements file with
comments. On thing is that the update to PyTorch 2.6.0 might cause some conflicts
and will have to be sorted out.

### Input larger than 512 token
> 📝 **Note:** The following section has been updated to use the latest
> transformers library as noted in the above section. To see prior results
> please look at the git history.

First run the original model with the [swa-prompt.txt](swq-prompt.txt) input file
to get a token count over 512:
```console
(venv) $ make embedding-run-original-model PROMPTS_FILE=swa-prompt.txt > original-model-output.txt
```

Then produce embedding on the current master branch:
```console
(venv) $ make embedding-run-converted-model PROMPTS_FILE=swa-prompt.txt > converted-model-master-output.txt
```

```console
(venv) $ tail converted-model-master-output.txt original-model-output.txt
==> converted-model-master-output.txt <==
embedding 530: -0.965656  1.360220 -0.225509  ...  1.683851 -1.329391  0.542756 
embedding 531: -2.035386  0.115241 -0.230382  ...  1.338015  0.129662  1.469703 
embedding 532: -0.303681  2.391175 -1.541024  ... -2.994880 -6.738236  2.204681 
embedding 533: -0.821824 -0.314456 -0.656939  ...  0.839123 -0.251856 -0.216256 
embedding 534: -6.953124  2.524179 -3.556372  ... -4.228094 -3.763176  1.713115 

Embeddings size: 410880
Saving logits to data/llamacpp-embeddinggemma-300M-embeddings.bin
Logits saved to data/llamacpp-embeddinggemma-300M-embeddings.bin
Logits saved to data/llamacpp-embeddinggemma-300M-embeddings.txt

==> original-model-output.txt <==
embedding 530: -0.951152  0.802068 -0.194350  ...  1.389915 -1.336990  0.639491 
embedding 531: -1.751817  0.208292  0.334419  ...  2.287567 -0.925609  0.149234 
embedding 532: -0.135422  1.639212 -1.219069  ... -1.935322 -5.207577  1.899551 
embedding 533: -0.760476 -0.438013 -0.575233  ...  0.915491 -0.058301 -0.046302 
embedding 534: -8.571507  3.587008 -3.270368  ... -5.668627 -3.859073  1.273620 

Total values: 410880 (535 tokens × 768 dimensions)

Saved bin embeddings to: data/pytorch-embeddinggemma-300M-embeddings.bin
Saved txt embeddings to: data/pytorch-embeddinggemma-300M-embeddings.txt
```
The output files are in the current directory. 

I'm working on adding this flag to be able to verify the logits with a specified
input flag like is used above.

A verification of the model can be run using:
```console
(venv) $ make embedding-verify-logits PROMPTS_FILE=swa-prompt.txt > logits-verification.txt
...
Loading model and tokenizer using AutoTokenizer: /home/danbev/work/ai/models/google/embeddinggemma-300M
n_tokens: 535
hidden_size: 768
pytorch embeddings:
[[-8.947923 11.567783 -3.949262 ... -7.488617 -4.011277  2.478082]
 [-1.131508 -0.050106  1.165388 ...  1.794341 -0.744414 -0.473869]
 [10.904386  0.457004  4.71285  ... -1.76934  -7.356499  3.377463]
 ...
 [-0.135422  1.639212 -1.219069 ... -1.935322 -5.207577  1.899551]
 [-0.760476 -0.438013 -0.575233 ...  0.915491 -0.058301 -0.046302]
 [-8.571507  3.587008 -3.270368 ... -5.668627 -3.859073  1.27362 ]]
llama.cpp embeddings:
[[-8.334854 10.894479 -4.177649 ... -9.079329 -4.938675  3.068214]
 [-2.335905 -0.645409  0.542473 ... -0.507136  0.170196  0.688234]
 [10.627003  0.560137  4.376873 ... -1.795107 -7.426371  3.062254]
 ...
 [-0.303681  2.391175 -1.541024 ... -2.99488  -6.738236  2.204681]
 [-0.821824 -0.314456 -0.656939 ...  0.839123 -0.251856 -0.216256]
 [-6.953124  2.524179 -3.556372 ... -4.228094 -3.763176  1.713115]]

...
4. Similarity Matrix Differences:
   Max difference: 0.9131
   Mean difference: 0.0350
   RMS difference: 0.0669

 === SUMMARY ===
 Average cross-model similarity: 0.9663
 Similarity matrix RMS difference: 0.0669
 ✅ EXCELLENT: Models are highly similar
```
The full output can be found in [logits-verification.txt](logits-verification.txt).

### Using SWA 2048
```console
(venv) $ make embedding-verify-logits PROMPTS_FILE=swa-prompt.txt > logits-verification-swa-2048.txt
...
n_tokens: 535
hidden_size: 768
pytorch embeddings:
[[-9.061672  9.337977 -3.009297 ... -5.05412  -4.892597  2.602074]
 [-2.566452  0.508311  1.705636 ...  3.844315  0.771225 -1.58529 ]
 [ 9.080079  0.52707   5.175964 ... -1.058788 -5.465713  3.505782]
 ...
 [ 0.264985  2.09919  -1.480251 ... -2.285295 -6.296007  2.017813]
 [-1.000267 -0.309579 -0.731549 ...  0.926902  0.065064 -0.034333]
 [-6.300206  3.647814 -4.840274 ... -3.138575 -4.10947  -0.52156 ]]
llama.cpp embeddings:
[[-9.110876  9.402618 -2.961948 ... -5.046772 -4.94622   2.539216]
 [-2.706887  0.463108  1.761616 ...  4.005383  0.780098 -1.67574 ]
 [ 9.265527  0.529064  5.293162 ... -1.174745 -5.64085   3.551831]
 ...
 [ 0.249931  2.075463 -1.460939 ... -2.261894 -6.241361  2.002948]
 [-0.991895 -0.308923 -0.732729 ...  0.931694  0.06267  -0.033718]
 [-6.238568  3.617653 -4.844738 ... -3.09892  -4.114971 -0.504128]]
...
```

Interestingly I notice the following in the output:
```console
(venv) $ head logits-verification-swa-2048.txt 
Modified sliding window: 257 -> 2048
Using unreleased model: None
Model class: <class 'transformers.models.gemma3.modeling_gemma3.Gemma3TextModel'>
Model file: transformers.models.gemma3.modeling_gemma3
Model's sliding_window: 2048
     2 -> '<bos>'
   818 -> 'The'
  3495 -> '▁concept'
   529 -> '▁of'
 16477 -> '▁artificial'
```

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


### Comparing swa masks with transformers

So perhaps we can simplify this a bit. Lets set the sliding window to say 6
and pass in a small sequence:

```console
(venv) $ make embedding-run-converted-model PROMPTS_FILE=small.txt
...
Input prompt: "Which planet is known as the Red Planet?"
Tokenized prompt (11 tokens): <bos>Which planet is known as the Red Planet?<eos>
output_reserve: reallocating output buffer from size 1.00 MiB to 11.03 MiB
print_mask: === Attention mask ===
print_mask: n_swa : 6, n_kv: 256, swq_type: LLAMA_SWA_TYPE_SYMMETRIC
print_mask: '0' = can attend, '∞' = masked
print_mask: Rows = query tokens, Columns = key/value tokens

      0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19
  0  0 0 0 0 0 0 0 0 0 0 0 ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞
  1  0 0 0 0 0 0 0 0 0 0 0 ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞
  2  0 0 0 0 0 0 0 0 0 0 0 ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞
  3  0 0 0 0 0 0 0 0 0 0 0 ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞
  4  0 0 0 0 0 0 0 0 0 0 0 ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞
  5  0 0 0 0 0 0 0 0 0 0 0 ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞
  6  0 0 0 0 0 0 0 0 0 0 0 ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞
  7  0 0 0 0 0 0 0 0 0 0 0 ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞
  8  0 0 0 0 0 0 0 0 0 0 0 ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞
  9  0 0 0 0 0 0 0 0 0 0 0 ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞
 10  0 0 0 0 0 0 0 0 0 0 0 ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞
print_mask: === Attention mask ===
print_mask: n_swa : 6, n_kv: 256, swq_type: LLAMA_SWA_TYPE_SYMMETRIC
print_mask: '0' = can attend, '∞' = masked
print_mask: Rows = query tokens, Columns = key/value tokens

      0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19
  0  0 0 0 0 ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞
  1  0 0 0 0 0 ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞
  2  0 0 0 0 0 0 ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞
  3  0 0 0 0 0 0 0 ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞
  4  ∞ 0 0 0 0 0 0 0 ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞
  5  ∞ ∞ 0 0 0 0 0 0 0 ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞
  6  ∞ ∞ ∞ 0 0 0 0 0 0 0 ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞
  7  ∞ ∞ ∞ ∞ 0 0 0 0 0 0 0 ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞
  8  ∞ ∞ ∞ ∞ ∞ 0 0 0 0 0 0 ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞
  9  ∞ ∞ ∞ ∞ ∞ ∞ 0 0 0 0 0 ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞
 10  ∞ ∞ ∞ ∞ ∞ ∞ ∞ 0 0 0 0 ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞
Embedding dimension: 768

embedding 0:  2.059559  2.318194 -1.411105  ... -0.787523 -0.218433  1.265910 
embedding 1: -1.726951 -0.802273  1.146937  ... -0.987142  0.446975 -0.152091 
embedding 2: -0.475775  0.721849 -1.177293  ... -0.617393 -1.896554 -0.878471 
embedding 3: -0.793247 -0.715338  0.158374  ... -0.664873 -0.686496 -0.362771 
embedding 4:  1.404825 -0.427954 -1.264170  ...  0.384742 -0.827241 -0.944257 
embedding 5: -0.088393 -0.564507 -0.049671  ...  0.723249 -0.172043  0.198801 
embedding 6: -1.928240  0.620383  0.298354  ...  0.394134 -0.452891  0.239224 
embedding 7: -3.681863 -3.136335  1.776549  ... -9.316843  2.508976  0.707935 
embedding 8:  1.404182  0.476730 -0.165926  ...  0.395325 -1.094332 -0.887429 
embedding 9: -0.266009 -3.047898  1.758015  ... -0.431742  2.205648 -1.889754 
embedding 10: -0.462513  2.652958 -0.623698  ...  1.057660 -2.112471 -1.306123
```
To visualize the mask in transformers we first need to clone the repo:
```console
(venv) $ git clone git@github.com:huggingface/transformers.git
(venv) $ cd transformers
(venv) $ pip install -e .
```
Then we add the printing of the mask in `modeling_gemma3.py`:
```console
$ git diff
diff --git a/src/transformers/models/gemma3/modeling_gemma3.py b/src/transformers/models/gemma3/modeling_gemma3.py
index d2ba04298d..89ecb52a0a 100644
--- a/src/transformers/models/gemma3/modeling_gemma3.py
+++ b/src/transformers/models/gemma3/modeling_gemma3.py
@@ -555,6 +555,8 @@ class Gemma3TextModel(Gemma3PreTrainedModel):
                 "full_attention": create_causal_mask(**mask_kwargs),
                 "sliding_attention": create_sliding_window_causal_mask(**sliding_mask_kwargs),
             }
+            from transformers.masking_utils import tensor_to_mask_visual
+            print(tensor_to_mask_visual(causal_mask_mapping["sliding_attention"][0][0], grid_size=(30, 50)))

         # embed positions
         hidden_states = inputs_embeds
```

```console
(venv) $ make embedding-run-original-model PROMPTS_FILE=small.txt
Modified sliding window: 512 -> 6
Using unreleased model: None
Model class: <class 'transformers.models.gemma3.modeling_gemma3.Gemma3TextModel'>
Model file: transformers.models.gemma3.modeling_gemma3
Model's sliding_window: 6
     2 -> '<bos>'
 24249 -> 'Which'
 13401 -> '▁planet'
   563 -> '▁is'
  3224 -> '▁known'
   618 -> '▁as'
   506 -> '▁the'
  4855 -> '▁Red'
 38342 -> '▁Planet'
236881 -> '?'
     1 -> '<eos>'
🀙🀙🀙🀙🀆🀆🀆🀆🀆🀆🀆
🀙🀙🀙🀙🀙🀆🀆🀆🀆🀆🀆
🀙🀙🀙🀙🀙🀙🀆🀆🀆🀆🀆
🀙🀙🀙🀙🀙🀙🀙🀆🀆🀆🀆
🀙🀙🀙🀙🀙🀙🀙🀙🀆🀆🀆
🀙🀙🀙🀙🀙🀙🀙🀙🀙🀆🀆
🀆🀙🀙🀙🀙🀙🀙🀙🀙🀙🀆
🀆🀆🀙🀙🀙🀙🀙🀙🀙🀙🀙
🀆🀆🀆🀙🀙🀙🀙🀙🀙🀙🀙
🀆🀆🀆🀆🀙🀙🀙🀙🀙🀙🀙
🀆🀆🀆🀆🀆🀙🀙🀙🀙🀙🀙
Hidden states shape: torch.Size([1, 11, 768])
All embeddings shape: (11, 768)
Embedding dimension: 768

embedding 0:  1.852479  2.278579 -1.399672  ... -0.868850  0.128775  1.163651
embedding 1: -1.783188 -0.789693  1.162982  ... -0.867609  0.346942 -0.057842
embedding 2: -0.115827  0.859096 -0.805346  ... -0.352525 -1.655916 -0.791396
embedding 3: -0.851912 -0.679230  0.012643  ... -0.637014 -0.664123 -0.606966
embedding 4:  0.627282 -0.026492 -2.292385  ...  0.153339 -1.051767 -1.041380
embedding 5: -1.110619  0.830399  0.568994  ... -0.125356 -0.008002  0.023205
embedding 6: -1.683385  0.482369  0.378439  ...  0.021804 -0.414762  0.176607
embedding 7: -4.531939 -3.586287  1.585973  ... -10.601041  2.521750  0.775239
embedding 8:  1.558994  0.495676  0.337419  ...  0.585594 -0.838154 -0.809764
embedding 9:  1.801445 -3.642905  1.861348  ... -0.203017  2.688000 -1.089822
embedding 10: -0.670791  3.021325 -0.776704  ...  1.235336 -1.902855 -1.526714

Total values: 8448 (11 tokens × 768 dimensions)
```
```
     0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
  0  0 0 0 0 ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞
  1  0 0 0 0 0 ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞
  2  0 0 0 0 0 0 ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞
  3  0 0 0 0 0 0 0 ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞
  4  ∞ 0 0 0 0 0 0 0 ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞
  5  ∞ ∞ 0 0 0 0 0 0 0 ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞
  6  ∞ ∞ ∞ 0 0 0 0 0 0 0 ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞
  7  ∞ ∞ ∞ ∞ 0 0 0 0 0 0 0 ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞
  8  ∞ ∞ ∞ ∞ ∞ 0 0 0 0 0 0 ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞
  9  ∞ ∞ ∞ ∞ ∞ ∞ 0 0 0 0 0 ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞
 10  ∞ ∞ ∞ ∞ ∞ ∞ ∞ 0 0 0 0 ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞ ∞
Embedding dimension: 768

embedding 0:  2.059559  2.318194 -1.411105  ... -0.787523 -0.218433  1.265910 
embedding 1: -1.726951 -0.802273  1.146937  ... -0.987142  0.446975 -0.152091 
embedding 2: -0.475775  0.721849 -1.177293  ... -0.617393 -1.896554 -0.878471 
embedding 3: -0.793247 -0.715338  0.158374  ... -0.664873 -0.686496 -0.362771 
embedding 4:  1.404825 -0.427954 -1.264170  ...  0.384742 -0.827241 -0.944257 
embedding 5: -0.088393 -0.564507 -0.049671  ...  0.723249 -0.172043  0.198801 
embedding 6: -1.928240  0.620383  0.298354  ...  0.394134 -0.452891  0.239224 
embedding 7: -3.681863 -3.136335  1.776549  ... -9.316843  2.508976  0.707935 
embedding 8:  1.404182  0.476730 -0.165926  ...  0.395325 -1.094332 -0.887429 
embedding 9: -0.266009 -3.047898  1.758015  ... -0.431742  2.205648 -1.889754 
embedding 10: -0.462513  2.652958 -0.623698  ...  1.057660 -2.112471 -1.306123
```
_wip_



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

