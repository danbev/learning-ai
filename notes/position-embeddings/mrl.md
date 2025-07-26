## Matryoshka Representation Learning
Matryoshka (the russion hidden dolls) is a way to train a model to learn representations of data in
a hierarchical manner. So lets say we have a "normal" model that learns a representation, an embedding
vector space of say 2048 dimensions. What this method allows is to also calculate a loss for smaller
sections of this 2048 dimensional space, 1024, 512, 256, 128, 64, 32, 16, 8. What this enables is that
we can take a 2048 embedding vector and slice it into smaller vectors and use them (provided that they
still enable the same similarity, more about this later when I've looked into it). So we might get as
input a 2048 dimensional vector but then slice it into a 256 dimensional vector which and use that for
lets say a query. This way we only need to store 256 embedding vectors instead of 2048 in our database.

_wip_
