## Sage Attention
This is a quantization method for the self-attention mechanism.

My understanding is that most quantization models focus on the linear layers of
the transformer model (the feed-forward layers) whereas this paper focuses on
the self-attention data/values. Actually there is FlashAttention 3 which uses
FP8 quantization for the self-attention mechanism but that is solely for the
Nvidia Hopper architecture. TODO: Read-up on FlashAttention 3.

Now with longer and longer sequence lengths and with the self-attension being
quadradic in complexity, as the sequence lenght increases so does the latency.

Just quantizing the Query, Key, Value, and P (the matrix representing the
softmax(QK^T/√d_k)) matrices to INT8 significantly degrades performance
according to the paper.

_wip_

Paper: https://arxiv.org/pdf/2410.02367v1
