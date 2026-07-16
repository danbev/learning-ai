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
