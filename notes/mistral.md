## Mistral 7B
Is an opensource model from Mistral AI:
https://mistral.ai/news/announcing-mistral-7b/

It used Grouped Query Attention, and Sliding Window Attention.

### Grouped Query Attention
TODO: Is this the same as explain in [transformer.md](./transformer.md)?

### Sliding Window Attention (SWA)
In the normal attention mechanism, the model attends to all tokens in the
sequence so as the sequence length increases the computation also increases 
quadratically. This is not feasible for long sequences. So the sliding window
attention is used to reduce the computation. The sequence is divided into

So in normal attention all the tokens attent to all other tokens in the sequence
but with a sliding window they only attent to the the tokens in the window. The
idea here is that the tokens inside of the window are more/most relevant.
Think about this as you are reading a book, if you on chapter 7 you are not
really concerned with attenting to works in chapter 1, they are probably not
relevant to the current chapter. On the other hand the words in the current
chapter are very relevant.
So this sounds like we are missing some interactions/attentions that might be
important?  
Not really actually, it turns out that because of the stacked layers in the
model and that the output of the prior layer is used as input to the next layer
and so on. 
This reduces the number of dot products needed to be calculated.

TODO: explain exactly how this works.
