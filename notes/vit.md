## Vision Transformer (ViT)
Transformer like architecture for vision tasks which has almost replaced CNN's
in the realm of computer vision (but who knows RNN/CNNs might be making a
comeback with RWKV and Mamba).

With transformers we take a sequence of text and split it into tokens, which are
then mapped to to embeddings using the LLM's vocabulary. 
For example, if we have the input sentence "Dan loves icecream". The first step
splits this into tokens, so we will have migth get 4 tokens:
```
["Dan", "loves", "ice", "cream"]
```
Next, these words are mapped to token id from the model's vocabulary:
```
 [1003]  [82]  [371]  [10004]
```
Now, the model will the take these input and map them into embeddings which
might be of a dimension of say 512. So we will have 4 vectors of 512 dimensions
```
'Dan'   1003  [0      ...        512]
'loves' 82    [0      ...        512]
'ice'   371   [0      ...        512]
'cream' 10004 [0      ...        512]
```

Lets take a look at how an image would be processed by an ViT. We start with
an image which is like our complete sentence above:
```
     +-----------------------------------------+
     |                                         |
     |                                         |
     |                                         |
     |                                         |
     |                                         |
     |                                         |
     |                                         |
     |                                         |
     |                                         |
     |                                         |
     |                                         |
     |                                         |
     +-----------------------------------------+
```
We split this into 16x16 squares (or patches):
```
     +-----------------------------------------+
     |16x16 |      |      |      |      |      |
     |      |      |      |      |      |      |
     +------+------+------+------+------+------+      
     |                                         |
     |                                         |
     |                                         |
     |                                         |
     |                                         |
     |                                         |
     |                                         |
     |                                         |
     |                                         |
     +-----------------------------------------+
```
So each of these 16x16 patches are like our tokens. Each one is then "flattened"
from a matrix into a vector:
```
       Im₁
     +------+
     |16x16 |  -> [0      ...        255]
     |      |
     +------+      

Im₁ = [0     ...        255]
Im₂ = [0     ...        255]
Im₃ = [0     ...        255]
...
Imₙ = [0     ...        255]
```
So at this point we have a sequence of vectors that represent the image. But at
this stage the entries are just the pixel values of the image, similar to how
tokens are just the split words/subwords of a sentence. In a transformer each
token is associated with an index in the models vocabulary. Using this index the
embeddings can be looked up in the models embedding table. This is what the
transformer model will use as the token embeddings. Also, the embeddings table
is something that is learned during training and can be thought of as a matrix
of weights. Keep that in mind for the next part.

Vision Transformers do not have a vocabulary in the traditional sense because
they deal with images, not discrete tokens. So, for an ViT we don't have a
vocabulary, but do generate embeddings, which are called patch embedding, and
they also do this using a matrix multiplication of a learned matrix. This looks
something like this:
```
E = VW * b

E = embedding for a single patch, like Im₁ for example
V = a flattened vector which contains raw pixel values of the patch
W = a matrix of weights
b = a bias term which is a vector of the same size as the embedding space.
```
The vector V (flattened patch) is multiplied by the matrix W (weights of the
linear projection), resulting in a new vector. This operation transforms the
patch from its original pixel space to the embedding space. The bias vector b
is then added to this result, producing the final embedding vector E.

