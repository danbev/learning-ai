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

### Implementation details
If we look at in the source code we can see that the generated file is loaded
like this:
```c++
    std::ifstream in(params.logits_file.c_str(), std::ios::binary);
    ...

    {
        char check[9]; check[8] = 0;
        in.read(check, 8);
        if (in.fail() || strncmp("_logits_", check, 8) != 0) {
            LOG_ERR("%s: %s does not look like a file containing log-probabilities\n", __func__, params.logits_file.c_str());
            return;
        }
    }
```
So first the file is opened and then the first 8 bytes are expected to contain
the magic string `_logits_`. The line defining check is first creating an array
of 9 characters on the stack, and then setting the last character to 0 the null
terminator. This is done to enable a proper string comparision incase the file
does no contain a string that is 8 characters long, so it is always a valid
C-style string.
The "header" format of the file is something like this:
```console
Bytes 0-7:   "_logits_"
Bytes 8-11:  n_ctx (4 bytes)
Bytes 12-15: n_vocab (4 bytes)
Bytes 16-19: n_chunk (4 bytes)
... etc
```
The the context length is read:
```c++
    uint32_t n_ctx;
    in.read((char *)&n_ctx, sizeof(n_ctx));
```
And then the vocabulary size and chunk size are read:
```c++
    int n_vocab;
    int n_chunk;
    in.read((char *)&n_vocab, sizeof(n_vocab));
    in.read((char *)&n_chunk, sizeof(n_chunk));
```
```console
(gdb) p n_ctx
$9 = 512
(gdb) p n_vocab
$10 = 262144
(gdb) p n_chunk
$11 = 576
```
After this the tokens are read from the file:
```conole
    std::vector<llama_token> tokens(size_t(n_ctx) * n_chunk);
    if (in.read((char *)tokens.data(), tokens.size()*sizeof(tokens[0])).fail()) {
```
So this will be a vector containing 512*567 tokens in this case:
```console
(gdb) p tokens.size()
$16 = 294912
```
So these are the tokens that were used by the base model when processed the
input text file.
```console
(gdb) p model.vocab.token_get_text(tokens[0])
$36 = 0x555556b98ba0 "<bos>"
(gdb) p model.vocab.token_get_text(tokens[1])
$37 = 0x5555574a0a68 "▁"
(gdb) p model.vocab.token_get_text(tokens[2])
$38 = 0x555556b99c08 "\n"
(gdb) p model.vocab.token_get_text(tokens[3])
$39 = 0x555556b9e5a0 "▁="
(gdb) p model.vocab.token_get_text(tokens[4])
$40 = 0x555556bf9298 "▁Robert"
(gdb) p model.vocab.token_get_text(tokens[4])
$35 = 0x555556bf9298 "▁Robert"
```
And if we inspect the file `wiki.text.raw` we can see that the first few lines:
```console
 = Robert Boulter = 
 
 Robert Boulter is an English film , television and theatre actor . He had a guest @-@ starring role on the television series The Bill in 2000 . This was followed by a starring role in the play Herons written by Simon Stephens , which was performed in 2001 at the Royal Court Theatre . He had a guest role in the television series Judge John Deed in 2002 . In 2004 Boulter landed a role as " Craig " in the episode " Teddy 's Story " of the television series The Long Firm ; he starred alongside actors Mark Strong and Derek Jacobi . He was cast in the 2005 theatre productions of the Philip Ridley play Mercury Fur , which was performed at the Drum Theatre in Plymouth and the Menier Chocolate Factory in London . He was directed by John Tiffany and starred alongside Ben Whishaw , Shane Zaza , Harry Kent , Fraser Ayres , Sophie Stanton and Dominic Hall . 
...
```
So those are the tokens part of the input file, but we also have the log
probabilities:
```c++
    std::vector<uint16_t> log_probs_uint16(size_t(n_ctx - 1 - n_ctx/2) * nv);
    std::vector<float>    kld_values(size_t(n_ctx - 1 - n_ctx/2)*n_chunk);
    std::vector<float> p_diff_values(size_t(n_ctx - 1 - n_ctx/2)*n_chunk);
    std::vector<float> logits;
    if (num_batches > 1) {
        logits.reserve(size_t(n_ctx) * n_vocab);
    }
    ...
    for (int i = 0; i < n_chunk; ++i) {
        const int start =     i * n_ctx;
        const int end   = start + n_ctx;

        const auto t_start = std::chrono::high_resolution_clock::now();

        if (in.read((char *)log_probs_uint16.data(), log_probs_uint16.size()*sizeof(uint16_t)).fail()) {
```
So this will process one chunk at at time and for each chunk it will read the
logits for that chunk from the file.

Then it will take a token from the tokens vector and decode it:
```c++

        // clear the KV cache
        llama_memory_clear(llama_get_memory(ctx), true);

        llama_batch batch = llama_batch_init(n_batch, 0, 1);

        for (int j = 0; j < num_batches; ++j) {
            const int batch_start = start + j * n_batch;
            const int batch_size  = std::min(end - batch_start, n_batch);

            // save original token and restore it after eval
            const auto token_org = tokens[batch_start];

            // add BOS token for the first batch of each chunk
            if (add_bos && j == 0) {
                tokens[batch_start] = llama_vocab_bos(vocab);
            }

            common_batch_clear(batch);
            for (int i = 0; i < batch_size; i++) {
                common_batch_add(batch, tokens[batch_start + i], j*n_batch + i, {0}, true);
            }

            if (llama_decode(ctx, batch)) {
                LOG_ERR("%s : failed to eval\n", __func__);
                llama_batch_free(batch);
                return;
            }

            // restore the original token in case it was set to BOS
            tokens[batch_start] = token_org;

            if (num_batches > 1) {
                const auto * batch_logits = llama_get_logits(ctx);
                logits.insert(logits.end(), batch_logits, batch_logits + size_t(batch_size) * n_vocab);
            }
        }
```
And then the logits generated from the quantized model will be passed to the
function `process_logits` with the logits from the original model:
```c++
        const int first = n_ctx/2;
        const float * all_logits = num_batches > 1 ? logits.data() : llama_get_logits(ctx);
        process_logits(n_vocab,
            all_logits + size_t(first)*n_vocab, // quantized models logits starting from the middle
            tokens.data() + start + first,      // tokens to predict, starting from the middle
            n_ctx - 1 - first,                  // number of predictions to make
            workers,
            log_probs_uint16,                   // base models stored log probabilities
            kld,                                // results accumulated here
            kld_ptr,
            p_diff_ptr);
```
Notice that this is only taking half of the context length and storing that in
`first`. This is then used with the with the starting index of the logits, so
it will skipping the first half of the logits/tokens. This is really about only
processing the second half as because when we have a causal model it can only
attend to tokens that are before the current token. Using the first tokens is
like a cold start and much more difficult for the model to predict. Doing it this
was allows the model to have a better chance of predicting the tokens and since
we do this same for the base and the quantized models this gives us a fair and
more realistic comparison.
```c++
static void process_logits(
    int n_vocab, const float * logits, const int * tokens, int n_token, std::vector<std::thread> & workers,
    double & nll, double & nll2, float * logit_history, float * prob_history) {
```
Notice that `nll` stands for negative log likelihood, which is what will be
passed to `exp` to get the perplexity value later. Recall that we are only
processing one token at a time so we have to accumulate the NLL values.
The `nll2` is the sum of the perplexity, and nll2 stands sum of sequared NLL values.
```c++
static void process_logits(
    int n_vocab, const float * logits, const int * tokens, int n_token, std::vector<std::thread> & workers,
    double & nll, double & nll2, float * logit_history, float * prob_history
) {
    std::mutex mutex;
    int counter = 0;
    auto compute = [&mutex, &counter, &nll, &nll2, logit_history, prob_history, n_vocab, logits, tokens, n_token] () {
        double local_nll  = 0;
        double local_nll2 = 0;
        while (true) {
            std::unique_lock<std::mutex> lock(mutex);
            int i = counter++;
            if (i >= n_token) {
                nll += local_nll; nll2 += local_nll2;
                break;
            }
            lock.unlock();
            const results_log_softmax results = log_softmax(n_vocab, logits + size_t(i)*n_vocab, tokens[i+1]);
            const double v = -results.log_softmax;
            local_nll += v;
            local_nll2 += v*v;

            logit_history[i] = results.logit;
            prob_history[i]  = results.prob;
        }
    };
    for (auto & w : workers) {
        w = std::thread(compute);
    }
    compute();
    for (auto & w : workers) {
        w.join();
    }
}

static results_log_softmax log_softmax(int n_vocab, const float * logits, int tok) {
    float max_logit = logits[0];
    for (int i = 1; i < n_vocab; ++i) {
        max_logit = std::max(max_logit, logits[i]);
    }
    double sum_exp = 0.0;
    for (int i = 0; i < n_vocab; ++i) {
        sum_exp += expf(logits[i] - max_logit);
    }
    return {logits[tok] - max_logit - log(sum_exp), logits[tok], expf(logits[tok] - max_logit) / (float) sum_exp};
}

struct results_log_softmax {
    double log_softmax;
    float  logit;
    float  prob;
};
```
And recall that the logits returned from llama_get_logits(ctx) and just the raw
unnormalized outputs from the final layer of the transformer. Softmax is used
to turn these into probabilites.
So the above is gathering all the probabilites needed for the summation in:
```console
Perplexity (PPL) = exp(-1/N * Σ log(p(x_i | x_<i)))

x_i = the actual token at position i
x_<i = all previous tokens (the context)
p(x_i | x_<i) = probability of token i given previous context
```
The `(log(p(x_i | x_<i)))` part is done by 
```c++
const results_log_softmax results = log_softmax(n_vocab, logits + size_t(i)*n_vocab, tokens[i+1]);
```
The negation is done by:
```c++
            const double v = -results.log_softmax;
```
Which is need because the log of a probability is always negative.

The summation is done by:
```c++
local_nll += v;
```
And notice that this is done after the negation as recall that the log of a
probability, a value between 0 and 1, is always negative.

It also takes care of the negation in the `log_softmax` function.

And the averaging (`1/N *`) is done by:
```c++
double mean = nll / count;
```
So this is processing one token at a time, calculating the log probability and
accumulating the negative log likelihood (NLL) values. The average NLL is then
calculated by dividing the total NLL by the number of tokens processed. This is
all done in "log-space" and later when all tokens have been processed we will
call exp which is exponentiating the average like this:
This is then passed to exp later in process_logits:
```c++
    const double ppl_val = exp(log_ppl.first);
```
