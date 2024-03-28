## Attention with Linear Biases Enables Input Length Extrapolation (Alibi)
This is a replacement of absolute position embeddings for transformers. Recall
that the extrapolation issue is when we have trained a model with a certain
sequence length, and then at inference time the input lengths exceed the maximum
length.

With absolut positional encoding the model learns the position during training,
remember that the position are added to the vector representation of the input
tokens (the embeddings). Because of this if the inference input exceeds the
training max length the model will often have trouble with the input and come up
with garbage output.

Recall that in the attention mechanism we have three matrices, the query, key,
and value matrices. The query and key matrices are multiplied together to
calculate the attention scores. So we have a query token and we want to figure
out how it relates to the values in the key matrix using dot products. 
This type of transformer is a causal transformer so the model can only attend
to tokens on the left, for before the current query token.

For example, for the second query token, the model can only attent to the first
token and the second and not the following tokens:
```
  q₁   q₂  q₃  q₄  q₅
     / |
    /  |
  k₁  k₂  k₃  k₄  k₅
```

So the model will take the dot product of the first token, followed by the
second token and so on:
```
  +-----+
  |q₁.k₁|                          q₁  q₂  q₃  q₄  q₅
  +-----+-----+
  |q₂.k₁|q₂.k₂|                    k₁  k₂  k₃  k₄  k₅
  +-----+-----+
```
In this case there is no information about the positions at all.
What is done in Alibi is we add the distance between the two tokens. For example,
q₂ is one position from q₁, so we add -1, and then multiply with a number m:
```
  +-----+             +-----+
  |q₁.k₁|             |  0  |
  +-----+-----+   +   +-----+-----+   *  m
  |q₂.k₁|q₂.k₂|       | -1  |  0  |
  +-----+-----+-----+ +-----+-----+-----+
  |q₃.k₁|q₃.k₂|q₃.k₃| | -2  | -1  |  0  |
  +-----+-----+-----+ +-----+-----+-----+
```
Notice that the futher back in context the token is the further away the token
is the smaller the number (since it is negative) and the more will be subtracted
from the attention value, that is the dot product between the query and the key.
So even if the result of a dot product is high, if the distance is large enough
the attention value will be low. This is the bias part of the Alibi, tokens that
are closer get a higher bias and tokens further away get a lower bias.

Positional embeddings in ALiBi are only applied after the query-key dot product:
```
 softmax(qᵢKᵀ + m * [0, -1, -2, -3, ...])
```
The scalar m is a head-specific value that is fixed before training.
