## Low-Rank Adaptation (LoRA)
LoRA is an example of Parameter Efficient Fine-Tuning (PEFT).

The motivation for LoRA is that most LLM's are too big to be fine-tuned on
anything except very powerful hardware, and very expensive to train from
scratch. So LoRA is a method that decomposes the original weight matrix into
smaller matrices which are called update matrices. The update matrices are then
used to create new layers that are then the only weights that are updated during
training.

The new layers are called the adaptation layers. This is possible by the fact
that large weight matrices are often low-rank matrices which means that it can
be approximated by the product of two smaller matrices.

Lets take a look at an example to clarify:
A matrix `W` of dimensions `m * n` and rank `r` can be written as the product
of two matrices `A` and `B`, where `A` has dimensions `m * r` and `B` has the
dimensions `r * n`.

```
W = A * B
    
      [2  4  6]             dim (m*n): 3x3, r (rank): 1
  W = [3  6  9] 
      [4  8 12]

  A = [2]                   dim: 3*1 (m*r)
      [3]
      [4]

  B = [1  2  3]             dim: 1*3 (r*n)

  [2]             [2  4  6]
  [3] [1  2  3] = [3  6  9]  
  [4]             [4  8 12]
       ↑              ↑
       |              |
     6 values      9 values
 
```
Recall that the rank is the number of linearly independent rows or columns in
a matrix. And notice that we haven't lost any information by decomposing the
matrix into two smaller matrices. We can reconstruct the original matrix by
multiplying the two smaller matrices together. While the above example shows
that we are using the same rank as the original matrix we can also use a lower
rank (not possible in this example as we already have rank of 1).

But one thing to note is that the `rank` is a parameter in LoRA and is something
that we can set ourselves, whereas in linear algebra this is something that is
a property of the matrix based on its element and this is fixed. This confused
me a lot initially as I was thinking about the rank as in the linear algebra
definition of rank. But the rank in LoRA is something that we can set ourselves.
We can think of the original matrix as a recipe for making a "fancy" dish. It
contains a lot of ingredients and steps. With the rank in LoRA we are saying
that we still want to make the same dish but we will use fewer ingredients and
steps. The number of ingredients and steps is the rank in LoRA.

Just to make one thing clear is that the rank in LoRA is often much smaller than
the rank of the matrix W. One way to think about this is that we are only
interested in slightly adjusting the features/weights of the base model and not
overhauling the entire model. We want to capture the adjustments for the new
tasks being trained/fine-tuned on so that it performs well on the new tasks. It
is beneficial if the base model is trained on similar things and not completely
different to get the best results.

So, now imagine that matrix W above is our weight matrix in a neural network.
This would be a lot larger in a neural network, but the concepts still applies.
We can decompose this matrix into two smaller matrices A and B.
Notice that instead of having 9 values we reduced that into 6 values in memory
which might not seem like a lot but when the matrix is very large this can make
a big difference.

So the idea here is that instead of re-training the entire model we can
decompose the weight matrix and train the smaller matrices instead and it will
have the same effect.

Initially `A` is initialized randomly using a gaussian/normal distribution.
Then `B` is initialized to 0. So the product of `A` and `B` is zero to begin
with.  Then `ΔW` is scaled by `α/r` where `α` is a scaling constant.

### Fine tuning
When we train a model from scratch the back propagation looks something like
this:
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

With fine-tuning we don't update the pre-trained weights, we only update the 
weights in the new layers that we add to the model. So it looks something like
this:
```
              +------------------+
              |   hidden layer   |        h = W₀x + ΔWx = W₀x + AB
              +------------------+
                      ↑       
                     W + α AB
               +------+---------------+
               |                      |
               |                +-----*-------+
               |                |     A*B     |
      +-------------------+  +-----------+ +-----------+   r = LoRA rank
      | pretrained weights|  | A weights | | B weights |
      |   Wₘ*ₙ (fixed)    |  |  Aₘ*ᵣ x   | | Bᵣ*ₙ x    |
      +-------------------+  +-----------+ +-----------+
                    ↑             ↑
                    |             |
                +------------------+
                |    x inputs      |
                +------------------+

α = alpha (scaling constant). A higher value means that low-rank matrices will
    have a greater impact on the original weight matrix. A lower value means
    that the changes introduces by the low-rank matrices will have a more subtle
    impact on the original weight matrix.
````
So we end up with two matrices because we have decomposed the original weight
matrix. Notice that matrix A has the dimensions `m * r` and matrix B has the
dimensions `r * n`, and multiplying them together gives us a new matrix with
the dimensions `m * n` which is the same as the original weight matrix.

The LoRA matrices can be merged (matrix addition) with the frozen weights which
does not increase the size of the model, which is one of the strengths of LoRA.

Other solutions like the "adapter" method would increase the size of the model
as the trained weights of the adapter are then taken and they extend the
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
but we still need to do matrix multiplication of the inputs and W, and then add
the results to A and B. 

### Matrix decomposition
It has been shown that large weight matrices can be described by smaller
matrices. So we can decompose a large matrix into smaller matrices and still
get the same, or close to the same, results. This is called matrix
decomposition.

Recall that the rank of a matrix is the number of unique rows or columns in the
matrix. If we have columns or rows that are linear combinations of other rows or
columns then we can remove them and still get the same results. This is what
the LoRA paper suggests and where the "low-rank" comes in LoRA comes from but
we must not confuse the rank of the weight matrix with the rank of the LoRA as
mentioned above.

### Inference time
During inference the base model can stay the same and the adaptation matrices
can be swapped. So it should be possible to have a single basemodel and
multiple adaptation matrices for different tasks. This would be a lot more
memory efficient than having multiple models for different tasks where result
of the A and B metrices are merged with the base model.

So the base model needs to first be loaded which could be done upon application
or container startup. The different lora models can be loaded as needed and
they can all share the same base model.

As an example of the number of parameters in lora, vs the number of parameters
in the base model:
```
lora trainable params: 2457600 
           all params: 3005015040 
            trainable: 0.08178328451893539
```
This is one of the strengths of LoRA. The number of trainable parameters is
much smaller than the number of parameters in the model. This means that it
should be possible to train the model on a smaller GPU than would be required
to train the model from scratch.

But do note that the same memory requirements are needed for inference plus
the memory requirements for the adaptation matrices if they are not merged with
the base-model. So the inference memory requirements will increase a little. But
this depends on whether the pre-trained model is merged with the adaptation
matrices or not. If the pre-trained model is merged with the adaptation matrices
then the memory requirements will not increase.

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
LoRA uses [Adam](./optimization-algorithms.md) optimization.

### LoRA Example
I usually prefer to have examples that run locally but for a lora example where
training/fine tuning is involved having access to GPU(s) is almost required.
I've got a Colab Pro account which gives me access to a GPU.
