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
