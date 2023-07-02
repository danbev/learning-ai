## Embedding Vector
When working with natura language models the input is text but neural networks
operate on matrixes/tensors of numbers, not text. Embedding vectors are a way
to turn text into vectors with the elements of the vector being numbers so that
text can be used with neural networks.

How this text to vectors of number is done can differ and one example is doing
[one-hot-encoding](./one-hot-encoding.md). Another options is using a count
based approach. And we also have the option to use embeddings which is what this
document will address.

Is a way of representing data like strings, music, video as points in a
n-dimension space. Doing this can allow similar data points to cluster together.

Word to vector (Word2Vec) was invented by Google in 2013 which takes as input
a word and outputs an n-dimensional coordinate, a vector. So, simlar words would
be closer to each other. Think of this as the tip of the vectors are in close
proximity to each other if the words are simliar.

For songs similar sounding songs would be nearby each other. And for images
simliar looking images would be closer to each other. This could be anything
really so we don't have to think just in terms of words.

How is this useful?  
Well, we can look at vectors/points close to get similar words, songs, movies
etc. This is called nearest neighbor search.

Embedding also allows us to compute similarity scores between these points
allowing us to ask how simlar is this song to another song. We can use the
Euclidean distance, the dot product, cosine distance, etc to calculate this
distance.

The learning stage is what positions these vectors close to each other.

We can use a one dimensional vector to represent each word, for example:
```
Why He She They Gras Tree
1   3   4   5    10   11

        She                  Tree
Why   He   they           Gras
|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|-
1  2  3  4  5  6  7  8  0 10 11 12 13 14 
```
This way we can calculate how similar to word are using:
```
Why and Gras:
10 - 1 = 9

He and she:
4 - 3 = 1
```
We can also define the encoding should have two values per vector making it
2 dimensional. So instead of each word being a single value it each one will
be a vector. We can use more dimensions as well and they can be .


### Universal sentence encoder
In this case entire sentences can be transformed into vectors which enables us
to do the same with sentences that we could with single words above.

