## Attention Sink
When Large language models (LLMs) exceed there context size they often shift out
old token embeddings from the kv-cache. Usually these are the oldest tokens in
the sequence (conversation). But it was noticed that this would then lead to
some models generating incoherent text.

Recall that the [attention](./attention.md) mechanism in LLMs is defined as follows:
```
Attention(Q, K, V) = softmax( (Q x Kᵗ)/√embedding_dim) x V
```
Now the focus here will be on the [softmax](./softmax.md) function which mandates
that all entries sum up to 1.0. The softmax function does not allow attention
scores to be exactly zero.


Take the following prompt:
```console
"Dan loves ice cream and goes to the store every day."
```
And lets say the raw attention scores are:
```console
Dan:   -5.0  (very low - not relevant)
loves: -5.0  (very low - not relevant)
ice:   -5.0  (very low - not relevant)
cream: -5.0  (very low - not relevant)
and:   -5.0  (very low - not relevant)
goes:  -3.0  (low - somewhat relevant)
to:    -3.0  (low - somewhat relevant)
the:   -2.0  (medium - more relevant)
store:  2.0  (high - very relevant for "every day")
every:  3.0  (highest - most relevant context)
```
After softmax this would yield:
```console
Dan:   0.01%  (tiny but not zero!)
loves: 0.01%  (tiny but not zero!)
ice:   0.01%  (tiny but not zero!)
cream: 0.01%  (tiny but not zero!)
and:   0.01%  (tiny but not zero!)
goes:  0.25%
to:    0.25%
the:   0.67%
store: 26.89%
every: 72.95%

Total: 100.00%
```
Now, something that is important to understand is that when the models are trained
they learn to recognize this type of attention pattern.

Lets say we shift out 5 entries from the cache because the context size has been
reached:
```
goes:  -3.0
to:    -3.0
the:   -2.0
store:  2.0
every:  3.0
```
And after the softmax function we get:
```console
goes:  2.48%
to:    2.48%
the:   6.69%
store: 26.89%
every: 61.46%

Total: 100.00%
```
What the model was trained to predict is something like "when I see ~73%
attention on 'every' and ~27% on 'store' with tiny bits distributed elsewhere,
predict 'day'".

Now the model sees "~61% attention on 'every' and ~27% on 'store' with different
distribution on remaining tokens" This is a completely different attention
pattern than what the model was trained on!
The model gets confused because:

* The attention percentages have shifted dramatically
* The overall "shape" of the attention distribution is different
* All those tiny 0.01% contributions that were spread across many tokens are now
  concentrated into fewer tokens

The StreamingLLM solution (the attention sink) is to keep the first four tokens:
```console
Dan:   0.01%  ← Keep
loves: 0.01%  ← Keep
ice:   0.01%  ← Keep
cream: 0.01%  ← Keep
store: 26.89%
every: 72.95%

Total: ~100.00%
```
Now the attention pattern matches what the model saw during training! The idea
is that the low scores that the model exepects to see are still there.
The tokens that are kept are called sinks, not like an I/0 sink but rather that
it sinks the attention scores to the expected values. This confused me a little
and I should rather think of this like a heat sink that absorbs heat to keep
the temperature stable. The attention sink absorbs the attention scores to keep
the attention distribution stable.

The actual content of these tokens is irrelevant, what matters is that they help
preserve the pattern that the model expects to see (what it was trained on)

It's purely about mathematical pattern matching:
* Training: Model learns "Pattern A" → Predict "day"
* Without sinks: Model sees "Pattern B" → Gets confused
* With sinks: Model sees "Pattern A" again → Predicts correctly

The early tokens serve as attention placeholders that maintain the statistical
signature the model expects to see.

Paper: https://arxiv.org/pdf/2309.17453
