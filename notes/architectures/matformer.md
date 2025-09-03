### MatFormer (Matryoshka Transformer)
The idea is simliar to the [mrl](../position-embeddings/mrl.md) and name
is inspired by the [Matryoshka dolls](https://en.wikipedia.org/wiki/Matryoshka_doll).

In the embedding case the idea was to train embeddings in a way that it becomes
possible to use smaller slices of the embedding if needed, but without having
any overhead in training.

In the transformer the Feed-Forward Network (FFN) block has a fixed
hidden/intermediate size. This layer has two linear transformations with a ReLU
activation function in between them (or some other activation function). The
first linear transformation expands the dimension of the input matrix, and the
activation (like ReLU or GELU) function is applied to each element of the matrix.
The second linear transformation reduces the dimension back to the original.
