## Attention
This document contains notes about different types of attention mechanisms used
in deep learning models like Attention (single-head), Multi-Head Attention,
Grouped Query Attention.

### Attention (single head)
Standard attention uses 3 matrices, a query matrix, a key matrix, and a value
matrix. 

Let's start with the following input sentence "Dan loves icecream". The first
step is to split this into tokens, so we will might get 4 tokens:
```
["Dan", "loves", "ice", "cream"]
```
Next, these words are mapped to token id from the model's vocabulary:
```
 [1003]  [82]  [371]  [10004]
```
Now, the model will the take these inputs and map them into vector embeddings
which might be of a dimension of say 512. So we will have 4 vectors of 512
dimensions:
```
'Dan'   1003  [0      ...        512]
'loves' 82    [0      ...        512]
'ice'   371   [0      ...        512]
'cream' 10004 [0      ...        512]
```
If the same word appears multiple times in the input, the same vector embedding
will be used for each occurance. So there is currently no context or association
between these words/token embeddings. They only contain information about each
word/token itself, and nothing about the context in which it appears. So the
word cold could mean that is is cold outside or that someone has a cold.

So with these embeddings the first thing the model does is to add a positional
encoding to each of the embeddings. In the original paper this used absolute
position encoding. I've written about this is [embeddings.md](./embeddings.md)
but RoPE could also be used but would be added to the Query and Key matrices
instead.

So we have our input matrix which in our case is a 4x512 matrix, where each
entry is one of the tokens in the input sentence. Notice that we in this case
have a sequence length of 4 (tokens that is). If we had a longer sequence this
work increase the size of the matrix. This has implications with regards to
memory usage and computation when the sequence lenghts get longer.

We take this matrix and make four copies, but we will only focus on the first
three for now:
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
Attention(Q, K, V) = softmax((Q x Kᵗ)/√dₖ) x V

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
So there is one row in Q for each token with a a dimension of 512.  And each
column in K represents one token (I usually think of matrix multiplication as
the rows in Q being the functions, and the columns in K being the inputs, so
we have 4 funtions in Q each taking 512 inputs from K).

So, lets just think about this for a second. We know we copied the input
matrix to Q and K. So I think we can visualize the attention score matrix like
this:
```       Dan  loves ice  cream
          (A)   (B)  (C)  (D)
          +-------------------+
Dan   (A) |AA | AB  | AC | AD |
          |-------------------|
loves (B) |BA | BB  | BC | BD |
          |-------------------|
ice   (C) |CA | CB  | CC | CD |
          |-------------------|
cream (D) |DA | DB  | DC | DD |
          +-------------------+
```
So AB is the dot product of the embeddings for word "Dan" and the embeddings for
the word "loves". Notice how we are "comparing" the word "Dan" with all the
other words in the sentence. And we do this for all words/tokens as well.

The dot product will give us some value that indicates how similar the two words
are (how far apart they are in the embedding space).
Once again I'd like to point out that if we had a longer sequence the QK matrix
would be larger.

In a causal attention model we would not want to attend to the future tokens
when calculating the attention scores. This is done by masking the future
tokens: 
```       Dan  loves ice  cream
          (A)   (B)  (C)  (D)
          +-------------------+
Dan   (A) |AA |     |    |    |
          |-------------------|
loves (B) |BA | BB  |    |    |
          |-------------------|
ice   (C) |CA | CB  | CC |    |
          |-------------------|
cream (D) |DA | DB  | DC | DD |
          +-------------------+
```

The next thing we do is we scale the values in the matrix by dividing them by
the square root of the embedding dimension. Recall that this called the
`scaled dot product attention` and this is the scaling part. This is done to
avoid stability issues if the dot product values are too large.

So we divide each value in the matrix with the square root of the embedding
dimension. After that we apply the softmax function to the matrix. This will
give us a matrix where the values are between 0 and 1 and the sum of the values
in each row will be 1. This matrix tells us how much attention we should pay to
each word in the input sentence. We take this matrix and multiply it with the
V matrix which is just the input matrix unchanged. The resulting matrix will
give us the attention scores for each of the words/tokens.

I just want to point out that this was using a single Q, K, and V matrix which
could be called `single head attention`. And also notice that there were no
learnable parameters in this case. We only used the input matrix which was
copied into Q, K, and V. In actual implementations what is used is something
called `multi-head attention` which I'll try to explain now.

### Multi-Head Attention (MHA)
This is the type of attention that was proposed in the original transformer
paper. 

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
updated by the model during training:
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

Query = (seq_len, embd_dim)
Key   = (seq_len, embd_dim)
Value = (seq_len, embd_dim)
W^q   = (embd_dim, embd_dim)
W^k   = (embd_dim, embd_dim)
W^v   = (embd_dim, embd_dim)
Q'    = (seq_len, embd_dim)
K'    = (seq_len, embd_dim)
V'    = (seq_len, embd_dim)
```
So the above is just a single head attention but this would be the same thing
if we have a multi-head attention but the number of heads is 1.

The `multi-head attention` function looks like this:
```
MultiHeadAttention(Q, K, V) = Concat(head_1, ..., head_h) x W^o
head_i = Attention(QW^qᵢ, KW^kᵢ, VW^Vᵢ)
Attention(Q, K, V) = softmax(Q, K, V) x V

h       = number of heads
d_model = embedding dimension size
dₖ      = d_model / h         For example 4 heads and d_model = 512, dₖ = 128
```
If we look at the Attention function it is the same as we saw earlier. What is
new is that we are going to split the matrices Q', K', and V' into smaller
matrices. This is the number of heads that we have. So the head part is just this
splitting of the matrices into smaller matrices. So why would we want to do this?
Well this is similar to how CNNs have multiple filters that learn specific
features during training. Each head will have its own learned weights which will
be specialized to learn different things about the input sequence.

So for example, if we want to have 4 heads and the embedding dimension size is
512, then we will have 4, 4x128 matrices. Each one of these are called a head and
they are separate from each other, and each one goes through the single-head
operation attention function that we went through above. 
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
seq_len  = 4
embd_dim = 512
heads    = 1

Query = (4, 512)
Key   = (4, 512)
Value = (4, 512)
W^q   = (512, 512)
W^k   = (512, 512)
W^v   = (512, 512)
Q'    = (4, 512)
K'    = (4, 512)
V'    = (4, 512)
```
So that would produce three matrices of size 4x512 if we have 1 head:
```
          512             512             512
     +-----------+    +-----------+    +-----------+ 
     |    Q'     |    |    K'     |    |    V'     |
 4   |           |  4 |           |  4 |           |
     |           |    |           |    |           |
     |           |    |           |    |           |
     +-----------+    +-----------+    +-----------+
```
But if we have two heads (h=2) we will split these into multiple matrices:
```
       256    256      256     256       256     256
     +-----+ +-----+  +-----+ +-----+  +-----+ +-----+
     | Q'_1| | Q'_2|  | K'_1| | K'_2|  | V'_1| | V'_2|
 4   |     | |     |  |     | |     |  |     | |     |
     |     | |     |  |     | |     |  |     | |     |
     |     | |     |  |     | |     |  |     | |     |
     +-----+ +-----+  +-----+ +-----+  +-----+ +-----+
```
Then we take each of these and perform the single head attention on each of
them separatly:
```
head1 = Attention(Q'_1, K'_1, V'_1) = softmax((Q'_1, K'_1, V'_1)/√dₖ) x V'_1
head2 = Attention(Q'_2, K'_2, V'_2) = softmax((Q'_2, K'_2, V'_2)/√dₖ) x V'_2
```
The we contatenate those heads and multiple by W^o.

Notice that we have `h` number of heads and each head has its own query, key,
and value matrices. So each query has a corresponding key matrix.

So for example, if we want to have 4 heads and the embedding dimension size is
512, then we will have 4, 4x128 matrices. Each one of these are called a head and
they are separate from each other, and are used to perform the single-head
attention function that we went through above. 
```
   +----------------+   +-------+     +--------+
   |      Q         |   |  Key  |     |   Q'   |
   |                | X |       |  =  |        |   
   |  (4, 512)      |   |(512,4)|     | (4, 4) |
   +----------------+   |       |     +--------+
                        |       |
                        |       |
                        +-------+
```

```
Attention(Q'₀, K'₀, V'₀) = softmax((Q'₀, K'₀, V'₀)/√dₖ) x V'₀
Attention(Q'₁, K'₁, V'₁) = softmax((Q'₁, K'₁, V'₁)/√dₖ) x V'₁
Attention(Q'₂, K'₂, V'₂) = softmax((Q'₂, K'₂, V'₂)/√dₖ) x V'₂
Attention(Q'₃, K'₃, V'₃) = softmax((Q'₃, K'₃, V'₃)/√dₖ) x V'₃
```
Those will output 4 (`sequence_length x dₖ`) matrices. So why would we want to do
this?  
Well, notice how each attention calculation will still be using all the words/
tokens of the input sequence but uses fewer dimensions than with the single head
attention. This has implication for the softmax calculation which now only sees
a subset of the embedding dimension values. This is what allows each of the
heads to "focus" on different parts of the dimension space and it is what
causes the model to learn different things about the input sequence.

These matrices are then concatenated into a single matrix:
```
                               +---------+
Concat(head_1, ..., head_h) =  |    H    |
                               | (4, 512)|
                               +---------+
```
And this matrix is then multiplied by a learnable parameter matrix W^o:
```
        +---------+     +-----------+    +-------+   (MH-A=MultiHead-Attention)
        |    H    |     |    W^o    |    | MH-A  |
        | (4, 512)|  X  | (512, 512)| =  |(4,512)|
        +---------+     +-----------+    +-------+
```
Notice that we did not have additional matrices in the single head attention
model.

In this case we have multiple matrices that have to be loaded into memory and
this can cause memory bandwidth issues.
Each attention head requires its own set of key and value matrices which have to
be stored in memory during the forward pass. So this is memory intensive and can
be a bottleneck for performance.

### Multi-Query Attention (MQA)
This was proposed in the paper "Fast Transformer Decoding: One Write-Head is
All You Need" and tried to address the memory bandwitdh shortcomings for
multi-head attention. Instead of having a key and value matrix for each head
we have only a single key and value matrix which is shared between all heads.
Fewer matrices means less memory to store intermediate.

In multi-head attention (MHA) we had multiple heads and each head has its own
query, key, and value matrices like we saw above:
```
             Multi-Head Attention
     +-----+   +-----+  +-----+   +-----+
     | Q'_1|   | Q'_2|  | Q'_3|   | Q'_4|
     |     |   |     |  |     |   |     |
     |     |   |     |  |     |   |     |
     |     |   |     |  |     |   |     | 
     +-----+   +-----+  +-----+   +-----+

     +-----+   +-----+  +-----+   +-----+   
     | K'_1|   | K'_2|  | K'_3|   | K'_4|
     |     |   |     |  |     |   |     | 
     |     |   |     |  |     |   |     | 
     |     |   |     |  |     |   |     | 
     +-----+   +-----+  +-----+   +-----+ 

     +-----+   +-----+  +-----+   +-----+
     | V'_1|   | V'_2|  | V'_3|   | V'_4|
     |     |   |     |  |     |   |     |
     |     |   |     |  |     |   |     |
     |     |   |     |  |     |   |     |
     +-----+   +-----+  +-----+   +-----+
```

In multi-query attention we still have the same number of query heads (h) but
only a single key and values vector:
```
             Multi-Query Attention
     +-----+   +-----+  +-----+   +-----+
     | Q'_1|   | Q'_2|  | Q'_3|   | Q'_4|
     |     |   |     |  |     |   |     |
     |     |   |     |  |     |   |     |
     |     |   |     |  |     |   |     | 
     +-----+   +-----+  +-----+   +-----+

                   +-----+
                   | K   |
                   |     |
                   |     |
                   |     |
                   +-----+

                   +-----+
                   | V   |
                   |     |
                   |     |
                   |     |
                   +-----+ 
```
The downside of this is that sharing the same K and V matrices between all
heads means that the model can't learn different things about the input sequence.
This is because the attention scores are calculated using the same
key and value matrices for all heads. This is a tradeoff between memory usage
and the ability to learn different things about the input sequence.

### Grouped Query Attention (GQA)
There is also a grouped query attention (GQA) where we have the same number of
query heads but we group them, and each group has its own key and value matrix.:
```
     +-----+   +-----+          +-----+   +-----+
     | Q'_1|   | Q'_2|          | Q'_3|   | Q'_4|
     |     |   |     |          |     |   |     |
     |     |   |     |          |     |   |     |
     |     |   |     |          |     |   |     | 
     +-----+   +-----+          +-----+   +-----+

          +-----+                     +-----+
          | K   |                     | K   |
          |     |                     |     |
          |     |                     |     |
          |     |                     |     |
          +-----+                     +-----+

          +-----+                     +-----+
          | V   |                     | V   |
          |     |                     |     |
          |     |                     |     |
          |     |                     |     |
          +-----+                     +-----+
```
Notice that GQA is a generalization of MHA and MQA:
* MQA is a special case of GQA where the number of groups is equal to 1.
* MHA is a special case of GQA where the number of groups is equal to the
  number of heads, for example 4 heads and 4 groups.

If we have 8 heads we can then have the following groups:
```
h = 8
8 groups, 4 groups, 2 groups, 1 group
h/n
8/1 = 8
8/2 = 4
8/4 = 2
8/8 = 1
```

#### Attention intuition
Visualize Q vector space as vectors in two dimensions and we have three vectors,
one for "I", one for "like", and one for "icecream". And we also have a vector
space for K with 3 vectors. When we calculate Q x Kᵗ we are getting a new square
matrix, and the values in this matrix contain the attention scores. What this is
doing is calculating the distances between the key matrix vectors to the query
vector (just one). This can be done by looking at the angle between the vectors
or calculating the dot product.

Smaller values in the attention score mean that we should pay less attention to
them and larger values mean that we should pay more attention to those tokens.

Next the output attention layer is passed to the Add&Norm layer which take it
as input and also takes a copy of the Value matrix which is passed around the
attention layer. This is what is called a skip connection or a residual
connection.

### Causal masking
We also have multi-head attention as described above in the decoder but there
is another layer called masked multi-head attention. This is while training,
well it is also used during inference but bare with me, where if we have a
translation task, the input to the decoder is the target sequence (the
translated version of the input sequence to the encoder). But we don't want the
decoders attention mechanism to take into account tokens that are ahead of the
current token. 
Lets say we are training a model and the input sequence is "Dan älskar glass"
which is Swedish for "Dan loves icecream" which is the target sequence which is
the input to the decoder. We don't want the computation of the first token `Dan`
to take into account any of the tokens ahead of it, like "loves", "ice", and
"cream". So we mask those tokens out. This is done by setting the attention
scores for those tokens to negative infinity. This will cause the softmax
function to output 0 for those tokens:
```       Dan  loves ice  cream
          +-------------------+
Dan       |0.9| ~inf|~inf|~inf|
          |-------------------|
loves     |0.3| 0.9 |~inf|~inf|
          |-------------------|
ice       |0.1| 2.3 |0.9 |~inf|
          |-------------------|
cream     |2.0| -1.5|0.2 |0.9 |
          +-------------------+
```
When performing inference the input to the decoder is the start of sequence
token and nothing else, and the decoder will generate the next token in the
sequence and add that to the input and continue like that. In this case there
no future tokens to mask but this is done for consistency (I think).

Encoders are used for classification and regression tasks.
Decoders are used for text generation tasks, like translation and summarization.

Encoder-Decoder are used for task like generative text like translation or
summarization. So if we wanted to train a model to translate a sentence from
English to Swedish, we would have the English sentence as input and the Swedish
sentence as the output. The model would then be trained to predict the Swedish
sentence given the English sentence.

Now, lets take a closer look at the boxes of the above diagram.
I've written about embeddings in [embeddings.md](./embeddings.md)
and positional encoding in [positional-encoding.md](./positional-encoding.md) so
lets skip them for now and start with the encoder layer.
So the next layer, or most often multiple layers, is the multi-head attention.
