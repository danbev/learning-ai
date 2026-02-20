## Medusa
This is a type of multi-token prediction but the underlaying model is not trained
for this. Instead this is slapped ontop an exising model. So the idea is to take
an existing model which is frozen and train small additional heads on top.

I came accros this when reading about MTP and just wanted to understand what this
is. The MTP heads are trained jointly with the main model.

In a standard Transformer, the "hidden state" h_t is the final high-dimensional
vector before it hits the lm_head to become a probability distribution over the
vocabulary.

Instead of running a whole new model, Medusa attaches several small MLP heads to
that final hidden state h_t.

```console
Head 0  Takes h_t -> Predicts x_{t+1}
Head 1: Takes h_t -> Predicts x_{t+2}
Head 2: Takes h_t -> Predicts x_{t+3}
```
Notice that the heads are all basing thier predictions on the same hidden state
h_t. So if the sequence was "Dan loves ice ", all three heads are trying to
predict the token that comes after the hidden space for "ice", and not the
token that comes after predicting x_t{t+1}. Head 1 is trying to guess the second
token without knowing what Head 0 picked for the first one. 

Medusa add heads, MLPs, to the last layer of the base model. An issue with Medusa
is that the heads are independent of each other, so they do not have any
information about the other tokens that are being predicted. This is in contrast
to the standard speculative decoding where the draft model can see the previous
tokens it has predicted and can use that information to make better predictions
for the next tokens.
