## Mixture of Depths
In this a type of transformer model where the attention and the feedforward
layer are computed with fewer tokens than the actual input sequence lentgh.
The motivation for this is to redude the computation of the attention which is
quadratic in the number of tokens.
The feedword layer takes the output of the attention layer and expands the
dimensionality, then operates on these expanded features, and reduces it back
to the original.

Now, the idea here is that not all input tokens are relevant and we might only
need some of them. By reducing the number of tokens, we can reduce the
computations for the attention and the feedforward layer.

So instead of passing all the input tokens to the attention layer only K tokens
are passed.
