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

Lets think about this a bit, normally we have a single lm_head which is a
matrix-vector multiplication that take the hidden state and performs a dot
product against each token embedding in the vocabulary. This gives a use a
similarity score of the current hidden state to each token in the vocab. So we
took an input token id, mapped it to an embedding vector which was then passed
through the layers of the model to give it context and peform other operation.
I imaging these operations as moving the vector around in the models hidden space
and after the last layer is where we then compare it with each token embedding
vector so see which are close to this hidden vector and the values are the
similarity scores, the result of the dot product.

So Medusa operates in logit space where we have a simlarity score for each token
in the vocab. This is a somewhat expensive operation to do and the more lm_heads
there are the more expensive it is.

In Medusa we will have multiple lm_heads:
```console
lm_head 0  Takes h_t -> Predicts x_{t+1}
lm_head 1: Takes h_t -> Predicts x_{t+2}
lm_head 2: Takes h_t -> Predicts x_{t+3}
```
Each lm_head is a separate weight matrix that was trained to predict the next
token (lm_head0) t+1. lm_head1 was trained to predict t+2, and lm_head2 was
trained to predict t_3 and so on. Which makes sense, like if they were the same
the would produce identical dot products.

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
