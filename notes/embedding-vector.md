## Embedding Vector
Is a way of representing data like strings, music, video as points in a
n-dimension space. Doing this can allow similar data points to cluster together.

Word to vector (Word2Vec) has invented by Google in 2013 which takes as input
a word and outputs an n-dimensional coordinate, a vector. So simlar words would
be closer to each other. Think of this as the tip of the vectors are in close
proximity to each other.

For songs simlar sounding songs would be nearby each other. And for images
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

### Universal sentence encoder
In this case entire sentences can be transformed into vectors which enables us
to do the same with sentences that we could with single words above.

