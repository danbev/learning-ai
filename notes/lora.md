## Low-Rank Adaptation (LoRA)
LoRA is an example of Parameter Efficient Fine-Tuning (PEFT).

The motivation for LoRA is that most LLM's are too big to be fine-tuned on
anything except very powerful hardware, and very expensive to train from
scratch.
So LoRA is a method that decomposes the original weight matrix into smaller
matrices which are called update matrices. The update matrices are then used
to create new layers that are and only the weights in these smaller layers are
updated during training.

The new layers are called the adaptation layers. This is possible by the fact
that large weight matrices are often low-rank matrices which means that it can
be approximated by the product of two smaller matrices.

Lets take a look at an example to clarify:
A matrix `A` of dimensions `m * n` and rank `r` can be written as the product
of two matrices `U` and `V`, where `U` has dimensions `m *r` and `V` has
the dimensions `r * n`.

```
W = A * B
    
      [2  4  6]             dim (m*n): 3x3, r (rank): 1
  W = [3  6  9] 
      [4  8 12]

  A = [2]                   dim: 3*1 (m*r)
      [3]
      [4]

  B = [1  2  3]             dim: 1*3 (r*n)

  [2]             [2  4 6 ]
  [3] [1  2  3] = [3  6 9 ]  
  [4]             [4  8 12]
       ↑              ↑
       |              |
     6 values      9 values
 
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

So the idea here is that instead of re-training the entire model we can
decompose the weight matrix and train the smaller matrices instead and it will
have the same effect.

Initially A is initialized randomly using a Gaussian/normal distribution.
Then B is initialized to 0. So the product of A and B is zero to begin with.
Then ΔW is scaled by α/r where α is a scaling constant.

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
              |   hidden layer   |        h = W₀x + ΔWx = W₀x + AB
              +------------------+
                      ↑       
                     W + AB
               +------+---------------+
               |                      |
               |                +-----*-------+
               |                |     A*B     |
      +-------------------+  +-----------+ +-----------+
      | pretrained weights|  | A weights | | B weights |
      |   Wₙ*ₙ (fixed)    |  |  Aₘ*ₙ x   | | Bₘ*ₙ x    |
      +-------------------+  +-----------+ +-----------+
                    ↑             ↑
                    |             |
                +------------------+
                |    x inputs       |
                +------------------+

````
So we end up with two matrices because we have decomposed the original weight
matrix. The LoRA matrices can be merged with the frozen weights with does not
increase the size of the model, which is one of the strengths of LoRA. Other
solutions like the "adapter" method would increase the size of the model as
the trained weights of the adapter are then taken and they extend the
pre-trained models' weights. But LoRA does not exclude other methods, they
could still be used in addition to LoRA.

So we have our frozen weights W:
```
      [2  4  6]             dim (m*n): 3x3, r (rank): 1
  W = [3  6  9] 
      [4  8 12]
```
And we have our matrices A and B which are the weights that we are training:
Now, lets modify them slightly:
```
  A = [2.1]                       dim: 3*1 (m*r)
      [3.3]
      [4.8]

  B = [1.0  2.2  3.6]             dim: 1*3 (r*n)
```
Now we will multiply A and B:
```
  [2.1]                   [2.1  4.62   7.56]
  [3.3] [1.0  2.2  3.6] = [3.3  7.26  11.88]
  [4.8]                   [4.8  10.56 17.28]
```
And then we merge the changes with the frozen weights:
```
      [2  4  6]   [2.1  4.62   7.56]   [4.1  8.62  13.56]
      [3  6  9] + [3.3  7.26  11.88] = [6.3  13.26 20.88]
      [4  8 12]   [4.8  10.56 17.28]   [8.8  18.56 29.28]
```
Notice that we are adding the pre-trained weights to the new weights.
So looking at that we are only updating the weights in the new layers A and B
but we still we still need to do matrix multiplication of the inputs and W,
and then add the results to A and B. But the computation of A and B which would
be done on GPUs is much less than the computation of W which would be done on
CPUs. So we are saving a lot of computation by only updating the weights in the
new layers I think. Remember that the above example is very very small and
the real weight matrices would be much larger.

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
During inference the base model can stay the same and the adaptation matrices
can be swapped. So it should be possible to have a single basemodel and
multiple adaptation matrices for different tasks. This would be a lot more
memory efficient than having multiple models for different tasks.

So the base model need to first be loaded which could be done upon application
or container startup. The different lora models can be loaded as needed and
they can all share the same base model.

As an example of the number of parameters vs the number of parameters in
the base model:
```
trainable params: 2457600 || all params: 3005015040 || trainable%: 0.08178328451893539
```
This is one of the strengths of LoRA. The number of trainable parameters is
much smaller than the number of parameters in the model. This means that it
should be possible to train the model on a smaller GPU than would be required
to train the model from scratch.

But do note that the same memory requirements are needed for inference plus
the memory requirements for the adaptation matrices. So the inference memory
requirements will increase a little. But this depends on whether the pre-trained
model is merged with the adaptation matrices or not. If the pre-trained model
are merged with the adaptation matrices then the memory requirements will not
increase.


### Singular Value Decomposition (SVD)
Is a linear algebra method that can be used to decompose a matrix into smaller
matrices. It is used in LoRA to decompose the weight matrix into smaller
matrices. The smaller matrices are then trained instead of the larger matrix.

Lets say we have matrix A and we want to decompose it into smaller matrices:
```
  A = UΣVᵀ
```
Where U and V are orthogonal matrices and Σ is a diagonal matrix. The diagonal

### Adam optimizer
LoRA uses Adam form model optimization.
TODO: explain Adam optimizer

### LoRA Example
I usually prefer to have examples that run locally but for a lora example where
training/fine tuning is involved having access to GPU(s) is almost required.
I've got a Colab Pro account which gives me access to a GPU.


