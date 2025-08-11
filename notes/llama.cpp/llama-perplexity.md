## llama-perplexity tool
This document contains notes about the `llama-perplexity` tool.

## Overview

1. Phase 1 - Generate baseline with unquantized model:
    - Run with --save-all-logits filename.kld to save tokens AND logits
    - Uses input text file to tokenize and generate logits for all tokens
    - Stores tokens and compressed log probabilities in the .kld file

2. Phase 2 - Compare with quantized model:
    - Run with --kl-divergence-base filename.kld (no input text needed)
    - Loads the stored tokens from the .kld file
    - Generates logits using the quantized model for the same tokens
    - Computes KL divergence between original and quantized model outputs

### Download the Wikitext-2 dataset
For running the perplexity evaluation, we can use the Wikitext-2 dataset. This
can be downloaded using the following command:
```console
    ./../../../scripts/get-wikitext-2.sh
```

### Generate logits for the Wikitext-2 dataset from base model
Using the converted base model, the non-quantized model, which will act as the
ground truth, we can then use it to compare with a quantized models:
```console
cmake --build ../../build --target llama-perplexity

mkdir -p ppl
OUTPUTFILE="ppl/$(basename $CONVERTED_MODEL).kld"

./build/bin/llama-perplexity -m $CONVERTED_MODEL \
    -f ppl/wikitext-2-raw/wiki.test.raw \
    --kl-divergence-base $OUTPUTFILE
```
This will generate and save both the tokens and the log probabilities to output
file. This is way in the next step we don't have to specify the input text
(wikitext-2-raw/wiki.test.raw) in this case again, but can just use the output
file.

### Perplexity output
This output of the perplexity evaluation can look something like this:
```console
====== Perplexity statistics ======
Mean PPL(Q)                   :  18.274923 ±   0.147295
Mean PPL(base)                :  19.087688 ±   0.155066
Cor(ln(PPL(Q)), ln(PPL(base))):  97.75%
Mean ln(PPL(Q)/PPL(base))     :  -0.043514 ±   0.001718
Mean PPL(Q)/PPL(base)         :   0.957419 ±   0.001645
Mean PPL(Q)-PPL(base)         :  -0.812765 ±   0.032991

====== KL divergence statistics ======
Mean    KLD:   0.171127 ±   0.000512
Maximum KLD:   8.847817
99.9%   KLD:   2.049528
99.0%   KLD:   0.881522
99.0%   KLD:   0.881522
Median  KLD:   0.129044
10.0%   KLD:   0.015355
 5.0%   KLD:   0.005119
 1.0%   KLD:   0.000561
Minimum KLD:   0.000002

====== Token probability statistics ======
Mean    Δp:  0.473 ± 0.024 %
Maximum Δp: 96.932%
99.9%   Δp: 53.430%
99.0%   Δp: 30.014%
95.0%   Δp: 14.910%
90.0%   Δp:  9.043%
75.0%   Δp:  2.252%
Median  Δp:  0.003%
25.0%   Δp: -1.340%
10.0%   Δp: -7.318%
 5.0%   Δp: -12.953%
 1.0%   Δp: -27.743%
 0.1%   Δp: -55.209%
Minimum Δp: -97.016%
RMS Δp    :  9.178 ± 0.043 %
Same top p: 78.637 ± 0.107 %
```

#### Mean PPL(Q) (Quantized Model)
This is the perplexity of the quantized model, and is a measure of how confused
the model is. The less confused/perplexed the model is, the better it is, so a
lower value is better.
```
PPL = exp(-1/N * Σ log(P(actual_token)))

PPL = perplexity
N = number of tokens
```
So this is summing all the log probabilities of the actual tokens that were
stored in the output file, and then dividing by the number of tokens. Then taking
the exponential of the result to get the perplexity value.

This is calculating the average negative log-likelihood of the correct tokens,
then exponentiating it.

#### Mean PPL(base) (Base Model)
This is the same as the previous value, but for the base model, which is our
non-quantized model. So most often the base model will have a lower value for
the perplexity but there are cases when the quantized model may have a lower
perplexity. For example, if the original model was Quantization-Aware Trained (QAT)
then if it was trained with a Q4 quantization, then a Q4_0 quantized model might
have a lower perplexity than the base model.


### Cor (Pearson correlation coefficient)
This is the Pearson correlation coefficient between log perplexities.
How consistently do both models find the same passages easy/hard?
```
Cor(ln(PPL(Q)), ln(PPL(base))):  97.75%
```
97.75% means they almost always agree on difficulty.

### Mean KLD
This is the mean KL divergence between the quantized model and the base model.
Formula:
```console
KL(P||Q) = Σ P(i) * log(P(i)/Q(i))
```
This is a measure of how different the two distributions are. The unit is in
natural logarithm bits.

### Maximum KLD
This is the worst single token prediction difference.


_wip_
