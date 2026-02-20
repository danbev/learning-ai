## Speculative Decoding

## Background/overview
So if we consider the "normal" or first speculative decoding we have a draft
model and the normal model. The draft model is much smaller and faster so it is
not expensive to make predictions with it, though it might not be as "good". But
many words don't require the prediction/model to be good either, filler words
like "the", "and", "is" etc are all words that the main larger model has to
predict one by one. If we can have the smaller model predict them, which are
mostly certain to also be what the large model will also predict we save and
utilize the main model better. But utilize I mean we can instead of predicting
a single token be predicting a batch of tokens, similar to how prefill/prompt is
done. Note that the base/normal/large model is not trained in a different way,
which will become a factor when we later look at other alternatives.

Having this additional model which has to be trained, need to be managed, like we
have a second model that has to be served and hooked up with the main large model,
share tokenizers etc is also a cost in maintenance. Medusa was therefor invented
to avoid the draft model and adds heads, MLPs, to the last layer of the model.


### Speculative decoding (standard)
The basic idea here is to have a smaller/faster model that acts as a "draft"
to predict the next few tokens, which are then verified by a larger model which
is the target model that we actually want to use.

Lets say we specify that we want the draft model to predict 5 tokens. It does
so in a normal autoregressive manner, and when it has predicted the 5 tokens
those tokens are passed to the target model including the original prompt tokens.

So we have an initial prompt just like I normally would to start things off. This
is passed to the draft model and it is set to predict 5 tokens, which it actually
samples and all.

These 5 tokens are then passed (including the tokens embeddings for the initial
prompt) to the target model. The target model will process these as it would
normally process a prompt, the additional 5 tokens at the end are just normal
prompt tokens as far as the target model is concerned.

And recall that the when a model processes a prompt it calculates the predictions
for every position simultaneously in a single pass.

```console
Drafted tokens: [A, B, C, D, E]
(5 cheap runs of the draft model)

The target model processes: [Prompt, A, B, C, D, E]
(1 expensive run of the target model)
```
Because the target model calculates logits for every position in parallel:
```console
It sees Prompt -> Predicts validation for A
It sees ...A   -> Predicts validation for B
It sees ...B   -> Predicts validation for C
It sees ...C   -> Predicts validation for D
It sees ...D   -> Predicts validation for E
It sees ...E   -> Predicts the brand new token F
```
It does all 6 of these calculations in that single forward pass.

Now, lets say that the prediction for token C mismatches what the draft model
predicted, lets say it predicted X. In that case we reject tokens C, D, and E
and output X as the next predicted token from the target model:
```console
[A, B, X]
```

### Self-Speculative decoding
So the above speculative decoding technique requires two models: a draft model
and a target which both need to loaded into VRAM. And these models also need to
share the same tokenizer/vocab.

Self-speculative decoding is a technique where the insight is that we don't need
all of the model layers to get a useful draft prediction, but can get away with
only using a few. So instead of processing the full say 42 layers of the model
it might exit early after say 8 layers to get a draft prediction. It takes the
output of the hidden state at layer 8, and applies a classification head.

### n-gram Speculative Decoding
n-gram in this context refers to a simplified "drafting" method that uses pattern
matching instead of a neural network to generate draft tokens. So this works by
looking backwards through the tokens that have already been generated (or the
initial prompt) and finding sequences of n tokens (n-grams) that have occurred
together previously. It then selects the prediction of those n-grams as the next
token(s) to draft.

In certains scenarious like coding where we often have repeated patterns this
can work quite well.
