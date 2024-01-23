## Flash Attention
This is a type of attention which uses information about the GPU hardware to
optimize the attention mechanism. 
Specifically [GPU](./gpu.md) hardware and memory transfers can hurt performance
where data is being copied to and from the GPU's SRAM to its HMB.

### Tiling
This involves restructuring algorithms to load block by block from GPU High
Bandwidth Memory (HBM) to the GPU SRAM to compute the attention. So the idea is
to load a block of data from the HBM to the SRAM. For the attention mechanism
the softmax presents a challenge, namely that it requires or couples the entire
row. 

First the matrices Q, K, and V are split into blocks which are the loaded from
HBM into the fast SRAM.

So how can the softmax function be computed in blocks when it requires the
entire row?  
Lets look at a concrete example:
```
X = [x₁, x₂, x₃]

First we compute the exponential of each element:
exp(x₁), exp(x₂), exp(x₃)

This is done so that we are guaranteed non-negative values and it also
amplicifies the differences between the values.

Then we sum the elements:
S = exp(x₁) + exp(x₂) + exp(x₃)

Then we divide each element by the sum:
Probability of x₁ = exp(x₁)/S
Probability of x₂ = exp(x₂)/S
Probability of x₃ = exp(x₃)/S

This would be simlar to calling the softmax function:
prop_x = softmax(X)
```
We can see that we needed to divide each element by the sum of all the elements
in the row. But if we have split this vector up into smaller blocks we won't
have access to all the values to compute the sum.

```
X = [x₁, x₂, x₃, x₄]

X¹ = [x₁, x₂]
X² = [x₃, x₄]
```
We compute the softmax just like we did above for each block, but we can't
simply concatenate the results as we need to take into account the values in
the other blocks. We need to adjust the calculations for each block to account
for their relative scale in the entire dataset.
So we need to find the max value in the entire vector:
```
m = max(x₁, x₂, x₃, x₄)
```
For X¹ we recompute each elements exponential eˣ¹⁻ᵐ, and eˣ²⁻ᵐ.
For X² we recompute each elements exponential eˣ¹⁻ᵐ, and eˣ²⁻ᵐ.

Then we compute the softmax for X¹:
```

### Reduce GPU memory I/O
Another things is that is done is that the attention matrix is not stored during
the forward pass so that it can be used in the backward pass. Instead it is
recomputed.

