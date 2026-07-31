## Linear attention
Standard attention uses softmax which links every token to every other token which
causes the cost of calculation to grow quadratically with the sequence length.
```console
output = softmax(QK^T/sqrt(d))V

Q is a matrix of shape [T x 128]. One token per row, T is tokens processed thus far.
K is a matrix of shape [T x 128], so K^T has the shape [128 x T].
V is a matrix of shape [T x 128].

Without softmax we get:
      (QK^T)        =   A    --> AV            = [Tx128]
[Tx128] x [128xT]     [TxT]     [TxT]x[Tx128]
```
Linear attention removes the softmax:
```console
output = (Q * ∅(K)^T) * V

∅ = simple activation function like Silu or just a normalization.
```
We can write this in a different way:
```console
output = Q * (∅(K)^T * V)
             [2D state S]

S = ∅(K)^T * V

      K^T V     =    S      --> Q S              = [Tx128]
[128xT]x[Tx128]  [128x128]     [Tx128]x[128x128]
```
`S` is a 2D matrix that acts as a state/memory that gets updated at each time step.

K transposed has T columns, where each column is a token's key vector k_t. And V
has T rows where each row is a token's value vector v_t:
```console
        [  |    |       | ]  [-- v_0 --]
K^T V = [ k_0, k_1, .. k_t]  [-- v_1 --]
        [  |    |       | ]  [    .    ]
                             [    .    ]
                             [    .    ]
                             [   v_t   ]
```
We can view this as a column by row matrix multiplication:
```console
S = K^T V = (k_0 v_0^T) + (k_1 v_1^T) + ... + (k_t v_t^T)
                ↑
                |
        [128x1]x[1x128] = [128x128]
```
So we are performing an outer product for each column in K with a row in V which
yields a [128x128] matrix. We then add all of them up. We still have T additions
but we never have to create a [TxT] matrix in memory. We only ever have to
matrialize one [128x128] matrix, which is added to the state matrix, also
[128x128] inplace. So we will have 128x128=16384 scalar multiplications and we
perform them T times. We also have 128x128=16384 element wise additions per set.
So we will have O(T * d²). 

Linear attention usually implies that we are looking at the whole sequence at a
time, while recurrence implies that we are processing one token at a time and
updating the state.

Plain linear attention has been around since about 2020, but it sufferred from
poor memory, it treated every token equally and just kept adding new information
to the state until it became a "blurry" mess of data:
```
S_t = S_{t-1} + (K_t * V_t^T)
```
What we will see is that linear attention addresses this by introducing gates:
```
S_t = forget(g1) * S_{t-1} + β(error correction)
```
