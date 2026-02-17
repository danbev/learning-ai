## Linear attention
Standard attention uses softmax which links every token to every other token which
causes the cost of calculation to grow quadratically with the sequence length.
```console
output = softmax(QK^T/sqrt(d))V
```
Linear attention removes the softmax:
```console
output = (Q * ∅(K)^T) * V

∅ = simple activation function like Silu or just a normalization.
```
We can write this in a different way:
```c++
output = Q * (∅(K)^T * V)
             [2D state S]

S = ∅(K)^T * V
```
`S` is a 2D matrix that acts as a state/memory that gets updated at each time step.

Linear attention usually implies that that we are looking at the whole sequence
at a time, while recurrence implies that we are processing one token at a time
and updating the state.

Plain linear attention has been around since about 2020, but it sufferred from
poor memory, it treated every token equally and just kept adding new information
to the state until it became a "blurry" mess of data:
```
S_t = S_{t-1} + (K_t * V_t^T)
```
What we will see is that Kimi Linear addresses this by introducing gates:
```
S_t = forget(g1) * S_{t-1} + β(error correction)
```
