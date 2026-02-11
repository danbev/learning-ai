## Kimi Linear

```c++
        case LLM_ARCH_KIMI_LINEAR:
            {
                llm = std::make_unique<llm_build_kimi_linear>(*this, params);
            } break;
```
And we can find the implementation in src/models/kimi-linear.cpp.

### Kimi Delta Attention (KDA)
This builds upon [mla](./mla.md) and the "Delta" part is what makes Kimi Linear
special. In older linear attention models, the memory matrix just kept adding
new information until it became a "blurry" mess of data.

KDA uses a Data-Dependent Update (The Delta Rule). Before adding new information
to the memory, it calculates how much of the existing memory is already similar
to the new input.

```
S_t = S_{t-1} + beta_t(v_t - S_{t-1} q_t) k_t

Where:
- S_t is the updated memory/state at time t
- S_{t-1} is the previous memory/state
- beta_t is a scalar (between 0 and 1) which acts like a write enable write enable
         switch. Similar to a forget gate in an LSTM.
- v_t is the value vector at time t (the current input's value representation)
- q_t is the query vector at time t (the current input's query representation)
- k_t is the key vector at time t   (the current input's key representation)
```
So this first multiplying the previous state (s_{t-1}) with the query (q_t) which
is the models prediction. S_{t-1} is everything that the models knows this far,
and q_t is the current query. By multiplying them together, we get a measure of
how much of the current query is already captured in the memory. This is the "similarity" or "overlap"

So we have S_{t-1} which is the current state which is like a key-value lookup
table that has been squashed into a single grid. But how can we "look up"
something in this grid, like we can just use [row][column] right. Instead the
"indices" are directions of the vectors.
```
      S_{t-1}        Q_t
 0  [0  ...   d]     [0]      [0]
    [0  ...   d]     [0]    = [0]
    [0  ...   d]     [0]      [0]
 d  [0  ...   d]     [0]      [0]
```
Lets say we have a d of size 4, and a sequence "The color is Red".
```
k = concept of the Color represented by a vector: [1 0 0 0] 
v = concept of Red represented by a vector      : [5 0 0 0]
```

In linear attention state a "key-value" pair is stored as an outer product of
v X k^T.  So looking at the above equation:
```
S_t = S_{t-1} + beta_t(v_t - S_{t-1} q_t) k_t
```
Lets start with the inner most expression (v_t - S_{t-1} q_t) where we perform
the multiplication of the current state with the query vector:
```

          [5 0 0 0]    [1]
S_{t-1}   [0 0 0 0] x  [0]
          [0 0 0 0]    [0]
          [0 0 0 0]    [0]
```
If we were to ask for the feature of index (0,0) the answer will be 5.
Now, imaging a new sequence "What is the color?" comes in. The query vector for
color could be [1 0 0 0] which is the same as the key vector for color.
```
          [5 0 0 0]  [1]    [5]
          [0 0 0 0]  [0]  = [0]
          [0 0 0 0]  [0]    [0]
          [0 0 0 0]  [0]    [0]
```
And notice that the resulting vector is our representation for "Red", so the
model was able to look it up.
Now, this is where the delta part comes into play. Imaging that the sequence
"The color is now Blue" comes in: 
```
v_t = "Blue" [0 7 0 0]
```
The model will has "Red" as its color (the feature in the state [5 0 0 0]).
The subtraction in (v_t - S_{t-1} q_t):
```
   [0]   [5]   [-5]   // delete Red
   [7] - [0] = [ 7]   // add Blue
   [0]   [0]   [ 0]
   [0]   [0]   [ 0]
```
This is the Delta part.

Beta is a vector of scalar values (0-1) that are often or perhaps always data
dependent. It could be computed as:
```
Beta_t = sigmoid(W_b x_t)
```
Assuming beta is 1 for simplicity, so this gives the model a way to gate, either
write-through (1) or not (0) the update for each feature. So this is an element
wise multiplication:
```
beta = [1 1 1 1] 

  [1]    [-5]     [-5]
  [1] *  [ 7]  =  [ 7]
  [1]    [ 0]     [ 0]
  [1]    [ 0]     [ 0]
```
So this enables selectively updating some features while leaving others unchanged.


And then we compute the outer product with the key vector k_t:
```
(Error) x k_t (color):

   [-5]               [-5 0 0 0]
   [ 7] x [1 0 0 0] = [ 7 0 0 0]
   [ 0]               [ 0 0 0 0]
   [ 0]               [ 0 0 0 0]

x = outer product operator.
```
And the last part is the addition:
S_t = S_{t-1} + Update
```
      [5 0 0 0]   [-5 0 0 0]   [0 0 0 0]
      [0 0 0 0] + [ 7 0 0 0] = [7 0 0 0]
      [0 0 0 0]   [ 0 0 0 0]   [0 0 0 0]
      [0 0 0 0]   [ 0 0 0 0]   [0 0 0 0]
```
Notice that the Red has been cancelled out, and the Blue has been written into
memory.

In Kimi Linear, by multiplying S_{t-1} by q_t, we are doing all of those lookups
at once in a single matrix-vector multiplication. The 2D matrix S has effectively
"pre-summed" all the past information, and q_t simply extracts what is relevant
right now.
