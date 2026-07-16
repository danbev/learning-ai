## Deepseek Sparse Attention (DSA)
So this addresses the issue that with increased context lengths calculating the
dot product accross all the tokens becomes computationally expensize and perhaps
not even possible. This is a technique to reduce the computations where as MLA
which was about reducing memory requirements. DSA compliments MLA and does not
replace it in any way. It is more like DSA can be performed prior to MLA.

The first stage of this is to reduce the number of tokens that we are going to
consider for attention. The idea is that not all tokens are actually relevant
and the first stage it to get attention scores in an efficient way and then only
use the most important ones (top-k). To make this as efficient as possible the
values will be quantized to 8-bits. We don't really care about the precision at
this stage, we just want to get a rough idea of which tokens are important, the
actual attention scores will be calculated in higher precision after this stage.
And it also does not need to be able to detect multiple features so there can
just be a single Key matrix. This state is called the "lightning indexer".

We have a multihead query matrix which is created by projecting the current
token embedding vector using a trained weight matrix W_Q.
```console
x shape [7168, 1]

Q_t_h_idx = x_t W_Q_h_idx

h = 64
d_idx = 16 or 32 dimensions
```

The key matrix is not multi-head but shared by all heads:
```console
K_s_idx = x_s W_K_idx

s = single shared key
```
Now, because we will be quantizing the projection there might be outlier spikes
in the data and standard 8-bit quantization either clips these large spikes or
squashes all the small values down to zero, which destroys precision.
```console
'Q_idx = FWHT(Q_t_h_idx)
'K_s_idx = FWHT(K_s_idx)
```
This used the Fast Walsh-Hadamard Transform (FWHT) to transform the values which
takes care of spikes in the data by smearing, or spreading them out amoung the
dimensions. This was the quantization in the next step will work.
TODO: include FWHT notes for other repo and link to them here.

We will then quantize the transformed values to 8-bits:
```console
Q_fp8 = quantize('Q_idx)
K_fp8 = quantize('K_s_idx)
```

```console
score(t, s) = Σ_h w_t_h * ReLU( Q_fp8[t, h] · K_fp8[s]^T )

w_t_h is a learned weight matrix that scales how much importance head h score
should carry.
```
What happens is that the indexer will compute the scores (attention scores) for
all past tokens (s = 1 .. N), and then sort the scores and selects the top k
tokens. And those tokens are then passed to the heavy MLA attention.

_wip_
