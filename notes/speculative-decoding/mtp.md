## Multi-Token Prediction (MTP)
This is from the following [Better & Faster Large Language Models via Multi-token Prediction](https://arxiv.org/pdf/2404.19737).

### Overview
Recall that during training we know the actual result of the predictions so
we can use that to verify/check how the model is doing. In normal next-token
prediction we predict one token at a time. In Multi-Token Prediction (MTP)
we predict multiple tokens at a time.

So if we have a sequence like "Dan loves ice", the next token predicted might be
"cream" which would be the t+1 token.
In MTP it will predict t+1, t+2, t+3, etc depending on how many tokens we want.

In normal next token prediction we have something like this:
```console

Target: "cream"
          ^
          |
    [Next-Token Head]  <-- Calculates logits for t+1
          ^
          |
    [ Shared Trunk  ]  <-- Transformer Layers
          ^
          |
Input: "Dan loves ice"
```

The normal loss function looks like this:
```console
L₁ = -∑ log P_∅(x_t+1 | x_t:1)
      t
```
So we have the negative sum of the log probability assigned to the actual next
token (x_t+1) given the input sequence (x_t:1). Theta being the model parameters.
So x_t+1 is the ground truth token we are trying to predict.

In MTP this will looks something like this instead:
```console

Targets:   "cream"       "cone"        "and..."      "cake"
           (t+1)         (t+2)         (t+3)         (t+4)
             ^             ^             ^             ^
             |             |             |             |
           [Head 1]   [Head 2]      [Head 3]      [Head 4]
             ^            ^             ^             ^
             |____________|_____________|_____________|
                               |
                               |
                          [ Shared Trunk ]  <-- Latent Representation z
                               ^
                               |
                       Input: "Dan loves ice"
```
So this is predicting "cream", "cone", "and...", "cake" all at once. How does
this benifit training?
The benifit is that the model learns faster so it needs fewer training tokens
to reach the same level of performance.
In standard training, the model only gets one "correction" (gradient signal) per
position: the immediate next word. In MTP, the model gets 4 corrections (if n=4)
for every position. This forces the shared "trunk" of the model to learn a
representation that is useful for the immediate future and the distant future
simultaneously.

The loss function for MTP:
```console
L_n = -∑ log P_∅(x_t+n:t+1 | x_t:1)
       t
```
`x_t+n:t+1` represents the entire sequence of future tokens at once. So if n=2
this would be "cream", "cone" (but token embeddings of course).
And the goal is to maximize the probability of getting this specific pair correct.
So to calculate multiple probabilities we multiply then togther P(A, B) which
can be written as P(A) * P(B|A).
```console
$("cream", "cone" | context) = P("cream" | context) * P("cone" | context, "cream")
```
But notice that to predict `P("cone" | context, "cream")` we need to already know
the value of "cream". And if we want to to this in parallel, which we do, we
need something else.
To make this work in parallel we assume that the hidden state z from the trunk
(by which they mean the body of the transformer architecture, everything before
the head(s)). So instead of predicting "cone" given "cream"  we are predicting
"cone" give the current context:
```console
P(x_t+n:t+1 | x_t:1) = P¹(x_t+1 | x_t:1) * P²(x_t+2 | x_t:1) * ... * Pⁿ(x_t+n | x_t:1)
```
Head 1 guesses t+1, Head 2 guesses t+2, etc all based on the same hidden state z.
The probability of all of them being right is the product of their individual
probabilities. And the log of a product is the sum of the logs, so we can add:
```console
log (P¹(..) * P²(..) * ... * Pⁿ(..))
```
And this becomes:
```console
log P¹(..) + log P²(..) + ... + log Pⁿ(..)
```
Then we put the negative sign in front to get the final loss function.
```console
L_n = -∑ [log P¹(x_t+1 | x_t:1) + log P²(x_t+2 | x_t:1) + ... + log Pⁿ(x_t+n | x_t:1)]
       t
```
Which we can write more compactly as by using the summation symbol for the
summing of logs inside the outer log sum:
```console
L_n = -∑ ∑ log P^i(x_t+i | x_t:1)
       t   i=1 to n
```

Now, recall that the output of the transformer layers/blocks, what the paper
calls trunk), is a single vector of floats often a size like 4096 or 8192 (the
hidden size). This is what the paper refers to as the latent representation z.
This how the transformer layers has moved the input tokens around in the
embedding space and we can think of this as a vector pointing to some location
in that space.
Next, we are going to multiply this vector by a matrix which has a shape of
[vocab_size, hidden_size] to get the logits for the next token prediction.
For example this might be [4096, 32000]:
```console
   0  [0 ... 4095]  <- token in the vocabulary
   1  [0 ... 4095]  <- token in the vocabulary
   2  [0 ... 4095]  <- token in the vocabulary
   ...

32000 [0 ... 4095]  <- token in the vocabulary
```
And we are going to multiply that with the vector:
```console
   0  [0 ... 4095]  [0]
   1  [0 ... 4095]  [1]
   2  [0 ... 4095]  ...
   ...              [4095]

32000 [0 ... 4095]
```
So this will multiply the vector against each embedding vector in in the matrix
and produce a new vector of size [vocab_size] which are the logits for each token.
And this is performing the dot product so the output is how similar is this
vector to the vector for a given token in the vocabulary. If the vectors point
in the same direction they will have a positive high value, if they point in
opposite directions they will have a large negative value.
So for each token in the vocabulary:
```console
Logit_cream = z . Vector_cream
```
For each token in the vocabulary we get a score (logit) and we get a high logit
(like 20.0) this means the vectors are pointing in almost the exact same
direction.
If the logit is zero this means the vectors are orthogonal (perpendicular).
The model is saying "This word has nothing to do with the current context."

And a negative Logit (like -10.0): The vectors are pointing in opposite
directions. The model is actively ruling this word out.

So, in MTP we will have multiple of these matrices.


Better reasoning:
By asking the model to predict t+4 ("cake") while it is still at t ("ice"), you
force it to "plan ahead." It cannot just guess "cream" because "cream" usually
follows "ice"; it must understand the context well enough to know that "cake" is
coming three steps later.

This benefits code generation and logic puzzles the most, where knowing the end
of the line is necessary to write the beginning correctly.


These additional heads, which are tensor weights that are trained are often
discared after training, or rather not included in the model used for inference.
But they can be used for speculative decoding which I'll take a closer look at
later in this document.

### Speculative Decoding with MTP
With MTP we don't need a separate smaller draft model and a larger one for
verifying it. Instead we can use the same model.
Flwo:
* Transformer blocks produce latent representation z
* Head 1 says "cream"
* Head 2 says "cone"
* Head 3 says "and"
We now have a draft sequence: "cream cone and".

The verification step is taking that whole sequence, "cream cone and", and feed
it back into the model again. This time just using a single head to see if "cone"
actually follows "cream", and "and" follows "cone", etc.

