## Transformers
Before the transformers architecture was developed, which was done by Google in
2017. Recursive Neural Networks (RNN) had been used up until this point but
transformers outperformed them in terms of translation quality and training
cost.

### Timeline
```
2017 Transformers
2018 ULMFit
2018 GPT
2018 BERT
2019 GPT-2 (not publicy released)
2019 RoBERTa
2019 DistilBERT (smaller version of BERT)
2019 BART
2020 GPT-3
2020 DeBERTa
2020 T5
2021 GPT-Neo
2021 GPT-J
```

### Backgound
Transformers are a type of neural network architecture that is used for
processing sequential data. They are used for tasks like translation, text
generation, and summarization.
So they can take text as input but neural networks don't process text, instead
the text need to be transformed into a format that the neural network can work
with. This is done by first tokenizing the text, and then these tokens are
converted into [embeddings](./embeddings.md) which are vectors of numbers that
represent the tokens. These embeddings are then fed into the neural network.


### Transformer Architecture Overview
One thing to keep in mind when looking at the "diagram" below is that it is used
while training a model and also when the model is used for inference. So during
training the weights and biases are updated and the finished result of training
is a model that can be used for inference (where the weights are just loaded
and not updated).

This destinction is particularly important when looking at the decoder as the
"input" to the decoder during training might be the target sequence (for example
 when doing translation would be the translated sentence in the target language)
, but during inference the input will be the start-of-sequence-token and this
sequence grows with each step as the model generates a new word.

```
[Encoder]

+-----------------------------+
| Input Sequence              |
| "Dan loves icecream"        |
+-----------------------------+
             |
             ↓
+-----------------------------+
| Tokenization & Embedding    |
+-----------------------------+
             |
             ↓
+-----------------------------+
| Positional Encoding         |
+-----------------------------+
             |
             ↓
+------------------------------+
| Encoder Layer 1              |
|            |                 |
|+-----------|                 |
||           |                 |
|| +---------+---------+       |
|| |Q        |K        |V      |
|| +-------------------------+ |
|| | Multi-Head Attention    | |
|| +-------------------------+ |
||            |                |
|+--+         |                |
|   ↓         ↓                |
| +-------------------------+  |
| | Add & Norm              |  |
| +-------------------------+  |
|             |                |
|+------------|                |
||            |                |
||            ↓                |
|| +-------------------------+ |
|| | Feed-Forward Network    | |
|| +-------------------------+ |
||            |                |
|+--+         |                |
|   ↓         ↓                |
| +-------------------------+  |
| | Add & Norm              |  |
| +-------------------------+  |
|   |Q        |K              |
|   |         |                |
+---|---------|----------------+
    |         +---------------------+
    +-----------------------------+ |
                                  | |
[Decoder]                         | |
                                  | |
+-----------------------------+   | |
| Outputs Sequence            |   | |
|                             |   | |
| Training: "Dan älskar glass"|   | |
| Inference: "<sos>"          |   | |      sos = start of sequence token
+-----------------------------+   | |
             |                    | |
             ↓                    | |
+-----------------------------+   | |
| Tokenization & Embedding    |   | |
+-----------------------------+   | |
             |                    | |
             ↓                    | |
+-----------------------------+   | |
| Positional Encoding         |   | |
+-----------------------------+   | |
             |                    | |
             ↓                    | |
+------------------------------+  | |
| Decoder Layer 1              |  | |
|            |                 |  | |
|+-----------|                 |  | |
||           |                 |  | |
|| +---------+---------+       |  | |
|| |Q        |K        |V      |  | |
|| +-------------------------+ |  | |
|| | Masked Multi-Head       | |  | |
|| | Attention               | |  | |
|| +-------------------------+ |  | |
||            |                |  | |
|+--+         |                |  | |
|   ↓         ↓                |  | |
| +-------------------------+  |  | |
| | Add & Norm              |  |  | |
| +-------------------------+  |  | |
|    |        +-------------------+ |
|+---|        |         +-----------+
||   |        |         |      |
||   ↓V       ↓Q        ↓ K    |
|| +------------------------+  |
|| | Multi-Head Attention   |  |
|| +------------------------+  |
||            |                |
|+--+         |                |
|   ↓         ↓                |
| +-------------------------+  |
| | Add & Norm              |  |
| +-------------------------+  |
|             |                |
|+------------|                |
||            |                |
||            ↓                |
|| +-------------------------+ |
|| | Feed-Forward Network    | |
|| +-------------------------+ |
||            |                |
|+--+         |                |
|   ↓         ↓                |
| +-------------------------+  |
| | Add & Norm              |  |
| +-------------------------+  |
|             |                |
|             ↓                |
| +-------------------------+  |
| |  Linear layer           |  |
| +-------------------------+  |
|             |                |
|             ↓                |
| +-------------------------+  |
| |  Soft max               |  |
| +-------------------------+  |
+------------------------------+
```

### Encoders
Are used for classification and regression tasks.

### Decoders
Are used for text generation tasks, like translation and summarization.

### Encoder-Decoder
Are used for task like generative text like translation or summarization.
So if we wanted to train a model to translate a sentence from English to
Swedish, we would have the English sentence as input and the Swedish sentence
as the output. The model would then be trained to predict the Swedish sentence
given the English sentence.


### Scaled dot product attention
Standard attention uses 3 martixes, a query matrix, a key matrix, and a value
matrix. 

Let's start with the following input sentence "Dan loves icecream". The first
step it split this into tokens, so we will have might get 4 tokens:
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
1003  [0      ...        512]
82    [0      ...        512]
371   [0      ...        512]
10004 [0      ...        512]
```
If the same work appears multiple times in the input, the same embedding will
be used for each occurance. So there is currently no context or association
between these works/token embeddings. The only contain information about each
work/token itself and nothing about the context in which it appears.

So with these embeddings the first thing in the model is to add a positional
encoding to each of the embeddings. In the original paper this used absolute
position encoding. I've written about this is
[vector-embeddings.md](./vector-embeddings.md).
So we have our input matrix which in our case is a 4x512 matrix, where each
is one of the tokens in the input sentence. We take this matrix and make 4
copies but we will only focus on the first three for now:
```
 +--------+          +-------+
 | Input  | -------> | Query |
 |        |          +-------+
 +--------+ -------> +-------+
                     |  Key  | 
                     +-------+
                     +-------+
            -------> | Value |
                     +-------+
                                               +--------------+
            -------------------------------->  | Feed forward |
                                               +--------------+
```
The attention function looks like this:
```
Attention(Q, K, V) = softmax((Q x Kᵗ)/√embedding_dim) x V

embedding_dim = 512 in our example
```
Lets start with looking at the nominator which is the matrix multiplication of
Q and K transposed:
```
   +----------------+   +-------+     +--------+
   |      Q         |   |  Key  |     |        |
   |                | X |       |  =  |        |   
   |  (4, 512)      |   |(512,4)|     | (4, 4) |
   +----------------+   |       |     +--------+
                        |       |
                        |       |
                        +-------+
```
So, lets just think about this for a second. We know we copied the input
matrix to Q and K. So I think we can visualize the attention score matrix like
this:
```       Dan  loves ice  cream
          (A)   (B)  (C)  (D)
          +-------------------+
Dan (A)   |AA | AB  | AC | AD |
          |-------------------|
loves (B) |BA | BB  | BC | BD |
          |-------------------|
ice (C)   |CA | CB  | CC | CD |
          |-------------------|
cream (D) |DA | DB  | DC | DD |
          +-------------------+
```
So AB is the dot product of the embeddings for word "Dan" and the embeddings for
the word "loves". Notice how we are "comparing" the word "Dan" with all the
other words in the sentence. And we done this for all words as well. The dot
product will give us some value that indicates how similar the two words are (
how far apart they are in the embedding space).

The next thing we do is we scale the values in the matrix by dividing them by
the square root of the embedding dimension. Recall that this called the
`scaled dot product attentions` and this is the scaling part. This is done to
avoid stability issues if the dot product values are too large.

So we divide each value in the matrix with the square root of the embedding
dimension. After that we apply the softmax function to the matrix. This will
give us a matrix where the values are between 0 and 1 and the sum of the values
in each row will be 1. This matrix tells us how much attention we should pay to
each word in the input sentence. We take this matrix and multiply is with the
V matrix which is just the input matrix unchanged. The resulting matrix will
give us the attention scores for each of the words/tokens.

I just want to point out that this was using a single Q, K, and V matrix which
could be called single head attention. And also notice that there were now
learnable parameters in this case. We only used the input matrix which was
copied into Q, K, and V. In actual implementation what is used is something
called multi-head attention which I'll try to explain now.

So we have our input matrix like before and we create copies of it just the
same as for single head attention:
```
 +--------+          +-------+
 | Input  | -------> | Query |
 |        |          +-------+
 +--------+ -------> +-------+
                     |  Key  | 
                     +-------+
                     +-------+
            -------> | Value |
                     +-------+
                                               +--------------+
            -------------------------------->  | Feed forward |
                                               +--------------+
```
But we also have additional matrices which are learnable, meaning they can be
updated by the model during training.   
```
 +--------+          +-------+    +-------+    +-------+
 | Input  | -------> | Query | X  | W^q   | =  |   Q'  |
 |        |          +-------+    +-------+    +-------+
 +--------+ -------> +-------+    +-------+    +-------+
                     |  Key  | X  | W^k   | =  |   K'  |
                     +-------+    +-------+    +-------+
                     +-------+    +-------+    +-------+
            -------> | Value | X  | W^v   | =  |   V'  |
                     +-------+    +-------+    +-------+
```

The multi-head attention function looks like this:
```
MultiHeadAttention(Q, K, V) = Concat(head_1, ..., head_h) x W^o
head_i = Attention(QW^qᵢ, KW^kᵢ, VW^Vᵢ)
Attention(Q, K, V) = softmax(Q, K, V) x V

h = number of heads
dₖ = d_model / h         For example 4 heads and d_model = 512, dₖ = 128
```
If we look at the Attention function it is the same as we saw earlier. What is
new is that we are going to split the the matrices Q', K', and V' into smaller
matrices. This is the number of heads that we have. So for example if we want
to have 4 heads and the embedding dimension size is 512, then we will have 4
4x126 matrices. Each one of these are called a head and the are separate from
each there are used to perform the single-head attention function that we went
through above. 
```
Attention(Q'₀, K'₀, V'₀) = softmax((Q'₀, K'₀, V'₀)/√dₖ) x V'₀
Attention(Q'₁, K'₁, V'₁) = softmax((Q'₁, K'₁, V'₁)/√dₖ) x V'₁
Attention(Q'₂, K'₂, V'₂) = softmax((Q'₂, K'₂, V'₂)/√dₖ) x V'₂
Attention(Q'₃, K'₃, V'₃) = softmax((Q'₃, K'₃, V'₃)/√dₖ) x V'₃
```
Those will output 4 (sequence_length x dₖ) matrices. So why would we want to do
this?  
Well, notice how each attention calculation will still be using all the words/
tokens of the input sequence but fewer dimensions than with the single head
attention. This has implication for the softmax calculation which now only sees
a subset of the embedding dimension values. It is this that allows each of then
heads to "focus" on different parts of the dimension space and it is what
causes the model to learn different things about the input sequence.

These matrices are then concatenated into a single matrix:
```
                               +---------+
Concat(head_1, ..., head_h) =  |    H    |
                               | (4, 512)|
                               +---------+
```
An this matrix is then multiplied by a learnable parameter matrix W^o:
```
        +---------+     +-----------+    +-------+   (MH-A=MultiHead-Attention)
        |    H    |     |    W^o    |    | MH-A  |
        | (4, 512)|  X  | (512, 512)| =  |(4,512)|
        +---------+     +-----------+    +-------+
```
Notice that we did not have additional matrices which we in the single head
attention model.


We start with our tokenized input, the embedded vector(s) that represents some
text, so in this example we assume that the input has already been through
BERT's preprocessing.

We can call this vector X, and lets say the dimension of this is 4 and that we
have two tokens:
```
          +--+--+--+--+
token 1   |  |  |  |  |
          +--+--+--+--+
token 2   |  |  |  |  |
          +--+--+--+--+
```

So we will have the following matrix multiplications:
```
       X                 W_Q              Q
  +--+--+--+--+      +--+--+--+
  |  |  |  |  |      |  |  |  |       +--+--+--+
  +--+--+--+--+      +--+--+--+    =  |  |  |  |     
  |  |  |  |  |      |  |  |  |       +--+--+--+
  +--+--+--+--+      +--+--+--+       |  |  |  |
                     |  |  |  |       +--+--+--+
                     +--+--+--+
                     |  |  |  |
                     +--+--+--+

       X                 W_K              K
  +--+--+--+--+      +--+--+--+
  |  |  |  |  |      |  |  |  |       +--+--+--+
  +--+--+--+--+      +--+--+--+    =  |  |  |  |     
  |  |  |  |  |      |  |  |  |       +--+--+--+
  +--+--+--+--+      +--+--+--+       |  |  |  |
                     |  |  |  |       +--+--+--+
                     +--+--+--+
                     |  |  |  |
                     +--+--+--+

       X                 W_V              V
  +--+--+--+--+      +--+--+--+
  |  |  |  |  |      |  |  |  |       +--+--+--+
  +--+--+--+--+      +--+--+--+    =  |  |  |  |     
  |  |  |  |  |      |  |  |  |       +--+--+--+
  +--+--+--+--+      +--+--+--+       |  |  |  |
                     |  |  |  |       +--+--+--+
                     +--+--+--+
                     |  |  |  |
                     +--+--+--+
```
Notice that we are taking the input matrix X and doing 3 linear transformations
using the same vector. So we will have one result for multplying it with the
query vector, one for the key vector, and one for the value vector. This will
produce 3 new vector spaces called the query space, the key space, and the value
space.

The Query vector is information that we are looking for, like the reason for
saying something when we speak. So since we multiply X with W_Q to get Q, the
Q vector has a context of sort, which in the case of Query is the information
about the token and how it relates to the overall goal of the sentence.

The Key vectors is the relevance of the word(s), represented by X, to the query. 
So the key vector how relevant this token is to the query.

Again since we mulitply X with W_K to get K, K also has a context which is the
relevance of X to the query.

The Value vector is just a compressed version of the embedded value. This is
called a context less version of the token because it does not have any
additional information apart from the token value itself.

So the Query and Key vectors endode context, that is how this token relates to
the entire sequence of tokens inputted to the model (this if after the encoders
have processed it). And again the Value vector is just a compressed version of
the token itself.

Scaled dot product attention:
```
scaleced_dot_product(Q, K, V) = softmax((Q x Kᵗ)/√embedding_dim) x V
```
We use the softmax so that all the values add up to 1 but still keep the
individual difference (not sure about the terminology here).

```

     Q               Kᵗ         Attention matrix
  
  +--+--+--+       +--+--+     +--+--+               Higher value for attention
  |  |  |  |       |  |  |     |  |  |
  +--+--+--+   x   +--+--+ =   +--+--+
  |  |  |  |       |  |  |     |  |  |
  +--+--+--+       +--+--+     +--+--+
                   |  |  |
                   +--+--+

  +--+--+        +--+--+                        Only done to make the values
  |  |  |        |  |  |                        smaller and faster to compute
  +--+--+ / √3 = +--+--+                        without loosing any information.
  |  |  |        |  |  |
  +--+--+        +--+--+

                 +--+--+
                 |  |  |
        softmax( +--+--+ )
                 |  |  |
                 +--+--+
                                V        
                 +--+--+    +--+--+--+     +--+--+--+
                 |  |  |    |  |  |  |     |  |  |  |
                 +--+--+  X +--+--+--+  =  +--+--+--+
                 |  |  |    |  |  |  |     |  |  |  |
                 +--+--+    +--+--+--+     +--+--+--+

```

Input sequence: "I like icecream"

The query vector, which recall is the embedding input vector multiplied by
W_q. Will this only a 

Visualize Q vector space as vectors in two dimensions and we have three vectors,
one for "I", one for "like", and one for "icecream".  And we also have a vector
space for K with 3 vectors. When we calculate Q x Kᵗ we are getting a new square
matrix, and the values in this matrix contain the attention scores. What this is
doing is calculating the distances between the key matrix vectors to the query
vector (just one?). This can be done by looking at the angle between the vectors
or calculating the dot product.

Smaller values in the attention score mean that we should pay less attention to
them and larger values mean that we should pay more attention to those tokens.

```
         ("like")       ("like")
         [0.23]         [0.22]
 softmax([0.87] /√dₖ) = [0.42]
         [0.70]         [0.36]

  total  (!= 1)         (== 1)

         [0.22]   [0.23 0.87 0.90 1.50]  ("I" value)
         [0.42] x [0.80 0.28 0.38 0.61]  ("Like" value)
         [0.36]   [1.10 0.56 0.43 0.88]  ("icecream" value)

```

Attention matrix:
 
```
                     +--+--+
 values for token 1  |  |  | 
 values for token 2  +--+--+
                     |  |  |
                     +--+--+
```


### Out of Vocabulary
This is part of transformers which enable them to tokenize words that are not
in their vocabulary. This is done by using a subword tokenizer where a word is
split into two parts where both are part of the vocabulary but the complete word
by itself.
For example: `tokenization` would be spit into `token` `##ization` where `##`
is a special token that indicates that the previous token is a subword of a
complete word.

### Add & Norm layer
This is a layer(s) that is used in transformers. It is used to normalize the
output of the attention layer. It is called Add&Norm because it adds the
output of the attention layer to the input (value matrix) and then normalizes
it. What is normalization? I've written about this in
[normalization.md](./normalization.md).

In the encoder, the first Add&Norm layer comes after the multi-head attention
layer. There are two inputs into this layer, the first is the original value
matrix which notice is passed around, this is called a residual connection or
a skip connection, the multi-head attention layer. The second input is the
output of the multihead attention layer. These are simply added together:
```
   v_mha = V + MHA

V = Value matrix.
MHA = Multi-Head Attention output
```
This is residual connection/skip connection.


So `v_mha` is a matrix where each row represents one transformed input token:
```
'Dan':   [0.1, 0.2, 0.3, 0.4]
'loves': [0.5, 0.6, 0.7, 0.8]
'ice':   [0.9, 1.0, 1.1, 1.2]
'cream': [1.3, 1.4, 1.5, 1.6]

D = 4 (is the dimension of the embeddings for each token)
```
Lets take one row, which is how layer normalization is performed:
```
Calculate the mean:
μ₀ = 1/D ∑ 0.1 + 0.2 + 0.3 + 0.4
μ₀ = 1/4 1 = 0.25

Calculate the variance:
σ²₀ = 1/D ∑ (0.1 - 0.25)² + (0.2 - 0.25)² + (0.3 - 0.25)² + (0.4 - 0.25)²
σ²₀ = 1/4 (-0.15)² + (-0.05)² + 0.05² + 0.15²)
σ²₀ = 1/4 (0.0225 + 0.0025 + 0.0025 + 0.0225)
σ²₀ = 1/4 (0.05)
σ²₀ = 0.0125

And then normalize:
   0.1 - 0.25
   ---------- = -1.3411
    √0.0125
```
So we do this for the entire first row which will produce the following values:
```
'Dan': [-1.3411, -0.4470, 0.4470, 1.3411]
```
By doing this for each token indepentently we make sure that the model looks at
the word `Dan` and understands it in its own right, not influenced by other
tokens. 

```
'Dan':   [-1.3411, -0.4470, 0.4470, 1.3411]
'loves': [-1.3411, -0.4470, 0.4470, 1.3411]
'ice':   [-1.3411, -0.4470, 0.4470, 1.3411]
'cream': [-1.3411, -0.4470, 0.4470, 1.3411]
```
These values are the same just because of the dummy values that I initialized
used and would normally differ.

To understand this a little better I think it might help to look at what
batch normalization would look like as well.
In this case we don't take each row but instead each column, starting with the
first one:
```
'Dan':   [0.1, 0.2, 0.3, 0.4]
'loves': [0.5, 0.6, 0.7, 0.8]
'ice':   [0.9, 1.0, 1.1, 1.2]
'cream': [1.3, 1.4, 1.5, 1.6]
```
So that the values produces for the first column will include values from
all the tokens, 'Dan' 0.1, 'loves' 0.5, 'ice' 0.9, and 'cream' 1.3:
```
Calculate the mean:
μ₀ = 1/D ∑ 0.1 + 0.5 + 0.9 + 1.3
μ₀ = 1/4 2.8
μ₀ = 0.7

Calculate the variance:
σ²₀ = 1/D ∑ (0.1 - 0.7)² + (0.5 - 0.7)² + (0.9 - 0.7)² + (1.3 - 0.7)²
σ²₀ = 1/4 (-0.6)² + (-0.2)² + 0.2² + 0.6²)
σ²₀ = 1/4 (0.36 + 0.04 + 0.04 + 0.36)
σ²₀ = 1/4 (0.8)
σ²₀ = 0.2

And then normalize:
   0.1 - 0.7
   --------- = -1.3416
    √0.2

   0.5 - 0.7
   --------- = -0.4472
    √0.2

   0.9 - 0.7
   --------- = 0.4472
    √0.2


   1.3 - 0.7
   --------- = 1.3416
    √0.2

'Dan':   [-1.3416]
'loves': [-0.4472]
'ice':   [0.4472]
'cream': [1.3416]
``` 
So in batch normalization, the tokens can influence each other, but this is not
the case in layer normalization as each token is handled completely separably.

### Feedforward layer
In the transformer acrhitecture, we have the multi-head attention layer,
followed by a Add&Norm layer, and then we have a feedforward layer.
This layer has two linear transformations with a ReLU activation function in
between them.

So lets say that the output from the Add&Norm layer is:
```
'Dan':   [0.1, 0.2, 0.3, 0.4]
'loves': [0.5, 0.6, 0.7, 0.8]
'ice':   [0.9, 1.0, 1.1, 1.2]
'cream': [1.3, 1.4, 1.5, 1.6]
```
This layer transforms the dimesion of the vector from 4 to lets say 8. So we
would have a weight matrix of size 4x8 which the above matrix would be
multiplied with creating a 8x8 matrix.

```
        (4x4)             FeedForward Weight Matrix 1 (4x8)
      +--+--+--+--+      +--+--+--+--+--+--+--+--+
Dan   |  |  |  |  |      |  |  |  |  |  |  |  |  |
      +--+--+--+--+      +--+--+--+--+--+--+--+--+
loves |  |  |  |  |      |  |  |  |  |  |  |  |  |
      +--+--+--+--+      +--+--+--+--+--+--+--+--+
ice   |  |  |  |  |      |  |  |  |  |  |  |  |  |
      +--+--+--+--+      +--+--+--+--+--+--+--+--+
cream |  |  |  |  |      |  |  |  |  |  |  |  |  |
      +--+--+--+--+      +--+--+--+--+--+--+--+--+
                                     |-----------|
                                           ↑
                                      new dimensions
```
An activation function is then applied to each element of the 4x8 matrix.
The increased dimension in the feedforward layer of a Transformer model is
achieved through a weight matrix, and the values in this matrix are learned
during the training process. These values become part of the model's learned
parameters (weights).

This is done so that the model can learn more complex relationships between the tokens.

This 4x8 matrix is then passed to a second linear transformation which usually
reduces the dimension back to 4. So this would be a 4x8 matrix multiplied by
a 8x4 matrix which would produce a 4x4 matrix.
```
         (4x8)                     FeedForward Weight Matrix 2 (8x4)
      +--+--+--+--+--+--+--+--+    +--+--+--+--+
Dan   |  |  |  |  |  |  |  |  |    |  |  |  |  |
      +--+--+--+--+--+--+--+--+    +--+--+--+--+
loves |  |  |  |  |  |  |  |  |    |  |  |  |  |
      +--+--+--+--+--+--+--+--+    +--+--+--+--+
ice   |  |  |  |  |  |  |  |  |    |  |  |  |  |
      +--+--+--+--+--+--+--+--+    +--+--+--+--+
cream |  |  |  |  |  |  |  |  |    |  |  |  |  |
      +--+--+--+--+--+--+--+--+    +--+--+--+--+
                                   |  |  |  |  |
                                   +--+--+--+--+
                                   |  |  |  |  |
                                   +--+--+--+--+
                                   |  |  |  |  |
                                   +--+--+--+--+
                                   |  |  |  |  |
                                   +--+--+--+--+
```
And this will result in a 4x4 matrix, the same size as before the feedforward
layer. This matrix has been transformed from the higher-dimensional space back
to the original dimensionality, but with values that have been processed through
both layers of the feedforward network.


#### CLS token
This is a token in the embedding which stands for classification. This is used
to represent the entire input sequence.
From ChatGPT4: 
```
It's important to note that despite the "classification" origin of its name, the
"CLS" token is used in all tasks where we need an aggregate representation of
the entire sequence, whether it's classification, named entity recognition, etc.

For example, if we're classifying a sentence as positive or negative
(sentiment analysis), we would feed the sentence into BERT, which would add the
"CLS" token to the start of the sentence. BERT processes the entire sequence and
outputs a vector for each token, but we would only use the vector corresponding
to the "CLS" token to make the positive/negative prediction.
```

Bert-small has 4 encoders and 15M learnable parameters.
Bert-base has 12 encoders and 110M learnable parameters.
Bert-large has 24 encoders and 340M learnable parameters.

### T5
Text-to-Text Transfer Transformer (T5) is a transformer model that is trained.
