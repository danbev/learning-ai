## Attention with Linear Biases Enables Input Length Extrapolation (Alibi)
This is a replacement of position embeddings for transformers. Recall that the
extrapolation issue is when we have trained a model with a certain sequence
length, and then at inference time the input lengths exceed the maximum length.
With absolut positional encoding the model learns the position during training,
remember that the position are added to the vector representation of the input
tokens (the embeddings). Because of this is the inference input exceeds the
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

So the model will take the dot product of the first token:
```
                                   
  +-----+
  |q₁.k₁|                          q₁  q₂  q₃  q₄  q₅
  +-----+-----+
  |q₂.k₁|q₂.k₂|                    k₁  k₂  k₃  k₄  k₅
  +-----+-----+
  
```
