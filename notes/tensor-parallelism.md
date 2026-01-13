## Tensor Parallelism (TP)
Similar to Pipeline Parallelism, Tensor Parallelism is another technique used when
we have models that are too large for a single GPU. But instead of splitting the
model's layers across the GPUs, we split the weight matrices. For example, lets
say we have to GPUs, GPU_A and GPU_B. We would store left have of the weigth
matrices on GPU_A, and the right half on GPU_B.
This means that both GPUs are involved in computing the same layers, but each
of them process only a part of the weight matrices. So both receive the same
input compute there part of the output and the combine the result which will be
the input to the next layer for the final output. This requires quite a bit of
communitcation as we need to send the input to both GPUs and then combine for
each layer, to the communication overhead can be significant and should be used
if the communitcation links are fast.

Flow token processing:
```
1. Input token is copied to both GPUs
2. Layer 1 computation:
   - GPU_A processes left half of weight matrix
   - GPU_B processes right half of weight matrix
   Both GPUs only have a partial sum at this point.
3. All-Reduce
   All GPUs combine their partial sums to get the complete output for Layer 1.
   Both now have the result from layer 1 and can proceed and use that as the input
   to layer 2.
```

And prompt processing works the same way but instead of a single token we have
many tokens but the same processing steps apply.

This is selected using the `--split-mode row` in llama.cpp
