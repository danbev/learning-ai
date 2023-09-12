## Low-Rank Adaptation (LoRA)
LoRA is an example of Parameter Efficient Fine-Tuning (PEFT).

The motivation for LoRA is that most LLM's are too big to be fine-tuned on
anything except very powerful hardware, and very expensive to train from
scratch.
So LoRA is a method decomposes the original weight matrix into smaller
matrices. The smaller matrices are then used to create new layers that are
layer and only the weights in these smaller layers are updated during training.
The new layers are called the adaptation layers. This possible by the fact
that large weight matrices low-rank matrices which means that it can be
approximated by the product of two smaller matrices.

For example:
A matrix `A` of dimensions `m * n` and rank `r` can be written as the product
of two matrices `U` and `V`, where `U` has dimensions `m *r` and `V` has
the dimensions `r * n`.

```
A = U * V
    
      [2  4  6]             dim (m*n): 3x3, r (rank): 1
  A = [3  6  9] 
      [4  8 12]

  U = [2]                   dim: 3*1 (m*r)
      [3]
      [4]

  V = [1  2  3]             dim: 1*3 (r*n)

  [2]             [2  4 6 ]
  [3] [1  2  3] = [3  6 9 ]
  [4]             [4  8 12]
```
Recall that the rank is the number of linearly independent rows or columns in
a matrix. And notice that we haven't lost any information by decomposing the
matrix into two smaller matrices. We can reconstruct the original matrix by
multiplying the two smaller matrices together.

So, now imagine that matrix A above is our weight matrix in a neural network.
This would be a lot larger in a nueral network, but the concepts still applies.
We can decompose this matrix into two smaller matrices U and V.
Notice that instead of having 9 values we reduced that into 6 values in memory
which might not seem like a lot but when the matrix is very large this can
make a big difference.

### Fine tuning
When we train a model from scratch it looks something like this:
```
      +------------------+
      |   hidden layer   |
      +------------------+
               ↑
               |
      +------------------+
      |    weights       |
      +------------------+
               ↑
               |
      +------------------+
      |     Inputs       |
      +------------------+

```

With fine-tuning we don't updated the pre-trained weights, we only update the 
weights in the new layers that we add to the model. So it looks something like
this:
```
              +------------------+
              |   hidden layer   |        h = Wx + WAWB
              +------------------+
                      ↑       
               +------+------+----------------+
               |             |                |
      +-------------------+  +-----------+ +-----------+
      | pretrained weights|  | A weights | | B weights |
      +-------------------+  +-----------+ +-----------+
                    ↑             ↑
                    |             |
                +------------------+
                |     Inputs       |
                +------------------+

````
Notice that we are adding the pre-trained weights to the new weights.
So looking at that we are only updating the weights in the new layers A and B
but we still we still need to do matrix multiplication of the inputs and W,
and then add the results to A and B. But the computation of A and B which would
be done on GPUs is much less than the computation of W which would be done on
GPUs. So we are saving a lot of computation by only updating the weights in the
new layers I think.

### Matrix decomposition
It has been shown that large weight matrices can be described by smaller
matrices. So we can decompose a large matrix into smaller matrices and still
get the same, or close to the same, results. This is called matrix
decomposition.

Recall that the rank of a matrix is the number of unique rows or columns in the
matrix. If we have columns or rows that are linear combinations of other rows or
columns then we can remove them and still get the same results. This is what
the LoRA paper suggests and where the "low-rank" comes in LoRA comes from.

### Inference time
During inference the basemodel can stay the same and the adaptation matrices
can be swapped. So it should be possible to have a single basemodel and
multiple adaptation matrices for different tasks. This would be a lot more
memory efficient than having multiple models for different tasks.
