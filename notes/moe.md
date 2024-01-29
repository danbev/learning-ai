### Mixture of Experts (MoE)
In this setup/architecture we have a number of neural networks, the experts,
which are trained together with a gating network.

### Gating Network
The gating network is not a transformer neural network like the experts but it
does take the same input, that is the input embeddings. The difference is that
the output of the gating network is the probability of which experts are most
suitable for the particular input.

Initially during training the weights of the gating network are random and the
backpropagation will adjust the weights that eventually the most suitable
experts are selected for the input (because they reduce the loss the most). So
a gating network might just be a simple linear or non linear layer followed by
a softmax.

And it is trained at the same time as the experts, or at least with the experts
in the system as it needs to calculate the loss of the outputs of the experts so
that its weights can be update accordingly.

So if we have 6 experts the output of the routing layer would be a vector of
six dimensions where each dimension represents the probability of the expert
being selected.

### Transformers and MoE
In transformers the experts are often the feed-forward layers. So we would have
the attention layer and normally followed by a feed-forward layer that operates
on all the tokens (rows of the output matrix from the attention layer), from the
output of the attention layer. This is called a dense feed-forward layer because
the layer handles all the tokens in the sequence.

So lets say we have an input token length (the input sequence as embeddings)
token length 4, embedding dimension 4096, and a hidden dimension of size 11008:
```
     0        4095
     [   ...    ]      // token embedding 1
 H = [   ...    ]      // token embedding 2
     [   ...    ]      // token embedding 3
     [   ...    ]      // token embedding 4
```
Each row of the matrix H is a token embedding. So the feed-forward layer will
operate on one of these at a time. 

So it will take expand from [1, 4096] x [4096, 11008] = [1, 11008], perform the
non-linear operation and then reduce it back to [1, 4096].

Now, 11008 specifies the number of dimensions, and each dimension holds a
value (parameter/weight). The size of this value can be Float32, Float16, or
a quantized value (like 8, 4, 2 bits).
So if we calculate the number of parameters as in 4096x11008 = 45,056,768. And
if we assume that the values are Float32 then each value will be represented by
4 bytes.

And we also have the matrix that reduces the dimensionality back to 4096. So
that is 11008x4096 = 45,056,768. So the total number of parameters in this
feed-forward layer is:
```
4096x11008 = 45,056,768
11008x4096 = 45,056,768
45,056,768 + 45,056,768 = 90,113,536
90,113,536 x 4 = 360,454,144 bytes
360,454,144 / 1024*1024 = 343.75MB
```

I hadn't thought about this before but in the feed-forward layer the tokens in
the sequence are passed through as individual tokens and they don't "interact"
with other tokens in the sequence. So they these operation could be different
feed-forward layers which could also be on different machines to distribute the
operations and memory requirements.

And if we recall from the [transformer](./transformer.md) document the
feed-forward layer consists of an operation that expands the dimensionality of
the input tokens and then performs a non-linear operation on the expanded
tokens, and then reduces their embedding back to the original.

Now, within a transformer architecture we can have a mixture of experts which is
a layer after the attention layer. The first thing in this layer will be the
gating layer which will calculate the probability of each experts, where experts
are just feed-forward networks/layers as described above.

Interestingly these feed-forward layers will become experts on things like
punctuation, verbs, nouns, etc. So a token that is a verb will be routed to the
expert that is an expert on verbs and so on.

