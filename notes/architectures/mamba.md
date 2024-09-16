## Mamba: Linear-Time Sequence Modeling with Selective State Space
Mamba is a type of selective state space model (SSSM), `sss` like a snake and
therefor named Mamba. I've got some notes on [SSM](./state-space-models.md).

Mamba is said that it might be just as influential as the transformer
architecture but this is left to be seen.

One of the authors is Tri Dao, was also involved in the developement of 
[Flash Attention](./flash-attention.md) and Mamba takes advantage of the GPU
hardware.

Transformers are effecient at training as they can be parallelized, incontrast
to RNNs which are sequential which makes training large models a slow process.

But, the issue with transformers is that they don't scale to long sequences
which is because the self attention mechanism is quadratic in the sequence
length. Every token has to attend to every other token in a sequence (n²). So if
we have 40 tokens that means 1600 attention operations, which means more
computation and this just increases the longer the input sequence is.

In this respect RNNs are more performant as they don't have the quadratic
scaling issue that the self attention mechanism has (but do have other
disadvantages like slower training).

The core of Mamba is state space models (SSMs). Before we go further it might
make sense to review [RNNs](./rnn.md) and [SSMs](./state-space-models.md).

Paper: [Mamba: Linear-Time Sequence Modeling with Selective State Space](https://arxiv.org/pdf/2312.00752)

Selective state space models, which Mamaba is a type of, give us a linear
recurrent network simliar to RRNs, but also have the fast training that we get
from transformers. So we get the best of both worlds.

### Mamba block diagram
```
        input
          |
          |-----------------------------+
          ↓                             |
     +-----------+                +------------+
    / projection  \              /  projection  \
   +---------------+            +----------------+
          |                            |
          ↓                            |
   +---------------+                   |
   | convolution   |                   |
   +---------------+                   |
          |                            |
          ↓                            ↓
   +---------------+            +----------------+
   |     Silu      |            |    Silu        |
   +---------------+            +----------------+
          |                            |
          ↓                            |
   +---------------+                   |
   |     SSM       |                   |
   +---------------+                   |
          |                            |
          ↓                            |
   +---------------+                   |
   | activation    |←------------------+
   +---------------+
          |
          ↓
   +---------------+
    \  projection /
     +-----------+
          |
          ↓
       output
```

### State Space Models

```
h_t = Āh_{t-1} + B̂x_t

Where:
A_bar = is the state transition matrix.
B_bar = input projection matrix.
x_t   = the input at time t.
h_t   = the hidden state at time t.
h_t-1 = the previous hidden state.
```

```
Input (x_t)
        |
        v
   +----------+
   |    B     |  Input Projection Matrix
   +----------+
        |
        v
    +---+---+
    |   +   | <---- A * h_{t-1}  (Previous Hidden State)
    +---+---+
        |
        v
   +----------+
   |  S4D/SSM | State Space Model
   +----------+
        |
        v
    +------------+
    |  LayerNorm |
    +------------+
        |
        v
    +---------+
    |   SiLU  |  Activation Function
    +---------+
        |
        v
 Hidden State (h_t)
```

One major difference with state space models is that they have state which is
not something the transformers have (well one might consider the kv-cache the
state). So transformers don't have an intrinsic state which gets updated as the
model processes a sequence. But neural networks like RNNs do have state, but
recall that they process the input sequentially.

To understand how Mamba fits in I found it useful to compare it to how
transformers look in an neural network:
```
Residual         ↑
     +---------> |
     |           |
     |   +-------------------+
     |   | Linear            |
     |   +-------------------+
     |           ↑
     |           |
     |   +-------------------+
     |   | Self-Attention    |
     |   +-------------------+
     |           ↑
     |           |
     |   +-------------------+
     |   | Normalization     |
     |   +-------------------+
     |           ↑
     |           |
     |           |
     +-----------+
                 |
                 ↑
```
And then we have Mamba:
```
Residual         ↑
     +---------> |
     |           |
     |   +-------------------+
     |   | Linear            |
     |   +-------------------+
     |           ↑
     |           |
     |   +-------------------+
     |   | SSM               |
     |   +-------------------+
     |           ↑
     |           |
     |   +-------------------+
     |   | Normalization     |
     |   +-------------------+
     |           ↑
     |           |
     |           |
     +-----------+
                 |
                 ↑

      SSNN (Selective State Neural Network)
```
So we can think of this as if we are swapping out the core layer but other
things stay pretty much the same.

### Selective State Space Models
Selective State Space is a type of state space, and a state space is defined
by two funcions:
```
h'(t) = Ah(t) + Bx(t)     (state equation)
yₜ = Ch(t) + Dx(t)        (output equation) (Dx is not referred to in the paper)

h ∈ Rⁿ  is the like the hidden state in an RNN
x ∈ R¹  is the input sequence, x(t) is the input at time t.
y ∈ R¹  is the output sequence
A ∈ Rⁿ×ⁿ is the state transition matrix
B ∈ R¹×ⁿ is the input projection matrix
C ∈ Rⁿ×¹ is the output matrix
```
Now, the current state of the system is give in `h(t)`. And the matrix A can
be thought of as rules that dictates how the state of the system should evolve
independently of the input.

Lets say we have the following sequence of inputs:
```
"Dan loves icecream"

h'(t) = Ah(0) + B["Dan"]
```
This is the first time so the hidden state is initialized to zeros. B["Dan"] is
the transformation of the input "Dan" by matrix B, which allows this new
information to be integrated into the model.

At the next timestep, we will have:
```
h'(t) = Ah(t-1) + B["loves"]
```
This time h(t-1) will contain information about "Dan" and it will be transformed
by applying matrix A. This reflects how the context of "Dan" evolves before the
next word "loves" is added. And this process then continues. I think what I did
not get initially was that we are "integrating/evolving" the hidden state with
h(t-1), which evolves the hidden state with the output of the previous
iteration.

Now, above we have the A, and b, which are continuous values as per the
definition of a state space model. This makes sense if we think about it as
this is not specific to neural networks or even computers. Think about an analog
system, for example an IoT device that reads the temperature from a sensor
connected to it. To process this signal it needs to be converted into digital
form. A simliar thing needs to be done in this case, as we can't use continous
signals with computers, just like an IoT device can't process an analog signal
directly. So we need to convert into descrete time steps, similar to how an
Analog-to-Digital Converter ([ADC]) would convert the signal into quantized
values. This step is called discretization in the state space model.

[ADC]: https://github.com/danbev/learning-iot/tree/master?tab=readme-ov-file#analog-to-digital-converter-adc

But we don't have a continous signal in this case but in this case we can think
of the inner state space as continous. Like a continous space that represents
a rich, unintrupted representation of information.

So instead of the using functions as shown above with concrete values we will
transform A and B into discrete values and the equations become:
```
     _       _ 
hₜ = Ahₜ₋₁ + Bxₜ
yₜ = Chₜ+ Dxₜ
```
To get the `Â` and `B̂` values a process called discretization is used.

### Discretization
For some background on this we can think of the internal state of the system
as a continous space of information. When we process tokens there are in a
descrete values. The A (state transition), B (input transition), and C
(output transition) matrices operate in the continuous space.


So we will first discretize the parameters A, and B of the state space model,
which means that we will convert them from continuous values to discrete values.

I think there are multiple methods/ways to do this but the paper mentions
the zero-order hold transform method which is a method for converting a
descrite time signal to continous time signal (the inner space). 

So instead of the using functions as shown above we concrete values we will
transform A and B into discrete values and the equations become:
```
     _       _ 
hₜ = Ahₜ₋₁ + Bxₜ
yₜ = Chₜ+ Dxₜ
```
Where `Â` and `B_hat` are:
```
Â = (I - Δ/2 A)⁻¹           (⁻¹ inverse bilinear transform)
B_hat = (I - Δ/2 A)⁻¹ ΔB        (⁻¹ inverse bilinear transform)

Δ = the time step, for example if we sample every minute then Δ = 1
I = the identity matrix
A = the state transition matrix
B = the input matrix
```

So we first transform the continuous parameters Δ, A, and B into discrete
parameters `Â`, and `B_hat`. This is done using forumlas:
```
Â = f_A(Δ, A)
B_hat = f_B(Δ, A, B)
```
Where `f_A` and `f_B` are the descretization functions/rules and can vary as I
understand it. The paper for example uses the Zero-Order Hold (ZOH) method.

Lets take a look at an example a recurrent computation of a descrete system:
```
h_0 = B_hat * x_0             // No Â * h_t-1 as this is the first time step
y_0 = C * h_0

h_1 = Â * h_0 + B_hat * x_1
y_1 = C * h_1

h_2 = Â * h_1 + B_hat * x_2
y_2 = C * h_2
```

Like me mentioned earlier this is great for inference as we only need to compute
one token at a time and the memory and computation is constant regardless of the
length of the input sequence. 
But at training we have the complete sequence already and having to go through
this sequencially is slow (escpecially compared to transformers which can take
large sequences in parallel).

So at this point we have seen a continuous time system (the original form), and
a discrete time (the one where we discretized the parameters A and B). But there
is also a third form namely a convolutional representation.

### Convolutional Representation

So this was really confusing to me. I think of an SSM as an RNN which processes
input/token sequentially. With a convolutional representation we have
a filter that is moved over the input and the dot product is computed. My
thought was how is this possible when it the input is sequential, like it can't
access future values so what is it convolving over?  
I think the answer is that the causual convolution where the filter is only
applied to past values. During training the model does have access to the
complete sequence but during inference it does not.

Recurrent formulation:
```
hₜ = Ahₜ₋₁ + Bxₜ                               (state)
yₜ = Chₜ                                       (output

h_0 = B_hat * x_0                              (state at timestep 0)
y_0 = C * h_0                                  (output at timestep 0)
```
Now we can rewrite the output as:
```
y_0 = C * h_0
    = C * B_hat * x_0            (h_0 = B_hat * x_0)
```
And we can then compute h1:
```
h_0 = B_hat * x_0
    = C * h_0
    = C * (B_hat * x_0)

h_1 = Â * h_0 + B_hat * x_1
               |
               +---+
                   ↓
h_1 = Â * (B_hat * x_0) + B_hat * x_1
y_1 = = C * h_1
    = C(Â (B_hat * x_0)                                __       _
    = C * Â * B_hat * x_0 + C * B_hat * x_1          (CABx_0 + CBx_1)

h_2 = Â * h_1 + B_hat * x_2
    = Â(Â * B_hat * x_0 + B_hat * x_1) + B_hat * x_2
    = Â^2 * B_hat * x_0 + Â * B_hat * x_1 + B_hat * x_2
y_2 = C * h_2
    = C(Â^2 * B_hat * x_0 + Â * B_hat * x_1 + B_hat * x_2)
    = C(Â^2 * B_hat * x_0) + C(Â * B_hat * x_1) + C(B_hat * x_2)
```
Now there is a pattern that emerges here is:
```
y_k = CÂ^k * B_hat * x_0 + CÂ^(k-1) * B_hat * x_1 + ... + C * B_hat * x_k
```
Now if we extract/split the input from the cooefficients we get:
```
K = Kernel (convoluational filter)

K_hat = (CB_hat, CÂ * B_hat, CÂ^2 * B_hat, ..., CÂ^(k-1) * B_hat)
```
So these are like the cofficients of the filter. If we arrange these into a
matrix we get:
```
K_hat = [CB_hat, CÂ * B_hat, CÂ^2 * B_hat, ..., CÂ^(k-1) * B_hat]
```
We can then calculate the output y using:
```
y = K_hat * x
```
Now, in the example above K has the same size of the input sequence lenght which
seems wrong as this is what we wanted to avoid. But this is only an alternative
way of representing the state space model which we can take advantage of during
training of models. We can parallelize the computation of the convolution.

In practice the kernel size K is fixed which might be something like 4.

```
Kernel (inverted):
 +------------------------------------+
 |  CÂ³B̂   |  CÂ²B̂   |  CÂB̂   |  CB̂   |
 +------------------------------------+

Input:
 +--------------------------------------------------------------+
 |  pad    |  pad    |  pad   |   x₀  |  x₁    |  x₂    |  x₃   |
 +--------------------------------------------------------------+

pad = 0

Process

 +------------------------------------+
 |  CÂ³B̂   |  CÂ²B̂   |  CÂB̂   |  CB̂   |
 +------------------------------------+
     ↓         ↓         ↓        ↓
 +--------------------------------------------------------------+
 |  pad    |  pad    |  pad   |   x₀  |  x₁    |  x₂    |  x₃   |
 +--------------------------------------------------------------+
     |        |         |        |
     +----------------------------
     ↓ 
 +------------------------------------+
 |  y₀     |         |        |       |
 +------------------------------------+
    y₀ = CB̂x₀ (all padding values are zero so they don't contribute)
```
And notice that `y₀` is the same as we calculated above for y₀.

Next, we slide the kernal forward one step:
```
           +------------------------------------+
           |  CÂ³B̂   |  CÂ²B̂   |  CÂB̂   |  CB̂   |
           +------------------------------------+
              ↓         ↓         ↓        ↓
 +--------------------------------------------------------------+
 |  pad    |  pad    |  pad   |   x₀  |  x₁    |  x₂    |  x₃   |
 +--------------------------------------------------------------+
              |        |         |        |
              +----------------------------
              ↓ 
 +------------------------------------+
 |  y₀     |  y₁     |        |       |
 +------------------------------------+
              y₁ = CÂB̂x₀ + CB̂x₁

```
And this can continue until we have processed the entire input sequence. Now
this look pretty sequential if we present it like this but if we look at how
we defined the coofficients of the kernel above.
For example we have our input and the kernel:
```
Input Sequence (x):
+----+----+----+----+
| x₀ | x₁ | x₂ | x₃ |
+----+----+----+----+

Kernel (K̂):
+-----+-----+-----+-----+
| CB̂  | CÂB̂ | CÂ²B̂| CÂ³B̂|
+-----+-----+-----+-----+
```
We can create a matrix of the input sequence like this:
```
Input Matrix (X):
+----+----+----+----+
| x₀ | 0  | 0  | 0  |            (t = 0)
+----+----+----+----+         
| x₁ | x₀ | 0  | 0  |            (t = 1)
+----+----+----+----+
| x₂ | x₁ | x₀ | 0  |            (t = 2)   
+----+----+----+----+
| x₃ | x₂ | x₁ | x₀ |            (t = 3)
+----+----+----+----+
```

And we can construct a the kernel matrix (a vector here for clarity) by
transposing the kernel:
```
Kernel Matrix (K̂ᵀ):
+-----+
| CB̂  |
+-----+
| CÂB̂ |
+-----+
| CÂ²B̂|
+-----+
| CÂ³B̂|
+-----+
```
We can then perform the above convolution using a single matrix operation:
```
+----+----+----+----+   +-----+   +----+
| x₀ | 0  | 0  | 0  |   | CB̂  |   | y₀ |
+----+----+----+----+   +-----+   +----+
| x₁ | x₀ | 0  | 0  | × | CÂB̂ | = | y₁ |
+----+----+----+----+   +-----+   +----+
| x₂ | x₁ | x₀ | 0  |   | CÂ²B̂|   | y₂ |
+----+----+----+----+   +-----+   +----+
| x₃ | x₂ | x₁ | x₀ |   | CÂ³B̂|   | y₃ |
+----+----+----+----+   +-----+   +----+

Output (y) = X × K̂ᵀ
```

In practice the kernel would be a matrix and not a vector like we used above.
Notice that in the forumalation of the ssm we have:
```
h_t = Ah_{t-1} + Bx_t
y_t = Ch_t
```
Where A, B, and C are the SSM coefficients.
If we look a Mamba model it may have a Kernel matrix of shape (4, 5120) which
is (kernel size, ssm_state_size).
The kernel matrix is a learned compact representation that encodes the necessary
information for the State Space Model (SSM) computation. Each row in this matrix
can be thought of as a compressed representation of the SSM coefficients for a
specific time step. So each row in the kernel is a vector, similar to how each
row in the input embedding matrix is a vector, the row represents a tokens and
the vector is the embedding for that token.

One thing to keep in mind is that the state h is intended to capture the history
of the sequence x. How this is done depends on the transformation matrices A
and B. In practice if the sequence is long then the model may forget earlier
information. The model prioritizes more recent information. Just to draw a
parallel to transformers, the self-attention mechanism can take the entire
sequence into account but it this can become very computationally expensive
as the sequence becomes very long.

So that is what is called the state space model, but we have not touched upon
the structured or selective parts of this yet. This is where S4 (structured state
space )comes in and it is defined as:
```
S4 = SSM + HiPPO + Structured Matrices
```
So we have SSM which is what we discussed above, then we have the addition
of HiPPO (History Preserving Operator?), and finally structured matrices.
Structured in this case means that the A matric has a rigid specific structure
which is intended to capture the long-term dependencies in the sequence.

The HiPPO operator looks like this and is a special variant, well actually it
specifies a way to construct the A and B matrices in a way that ensures that
a model can retain a high-resolution of past inputs.
```
x' = Ax + Bu
```
In the HiPPO framework, the design of matrix A is crucial for determining how
the internal state evolves to preserve historical information. The matrices A
and B are called HiPPO matrices. As we mentioned above the matrices A and B are
learned during training and for the HiPPO matrices this is done by using special
algorithms.

HiPPO aims to optimize A to ensure that older inputs are gradually and smoothly
"compressed" into the model's state, without being abruptly forgotten. So A is
the transition from h(t) to h(t+1) and note that this is not dependent on the
current input token (u or x, whatever the name of the thing following B is).

Similarly, the HiPPO approach influences the design of matrix B, which governs
how new inputs are incorporated into the model's state. The goal is to integrate
new information in a way that complements the historical data already
represented within the model's internal state.

Recall that this is a mapping of the input u into the state space x (I know that
I'm using x as the state space where above I used h(t), and also using u as the
input. I've seen both of these ways of naming). The idea is to design a state
the can capture the inputs entire history.

One question that was "asked" was, "using the current state, `x_t`, can we
reconstruct the history of inputs?"

HiPPO operator:
```
x'(t) = Ax(t) + Bu(t)
```

HiPPO matrix

```
      { 0     n < k }
Aₙₖ = { n+1   n = k }
      { 2n+1  n > k } 

n = row index
k = column index
```
Lets say we have the following matrix:
```
         0  1  2  3  4
row 0  [ 1, 2, 3, 4, 5]
row 1  [ 1, 2, 0, 0, 5]
row 2  [ 1, 2, 3, 0, 5]
row 3  [ 1, 2, 3, 4, 5]
row 4  [ 1, 2, 3, 4, 5]
````
And if we start with the n < k condition:
```
n < k

         0  1  2  3  4
row 0  [ 1, 0, 0, 0, 0]
row 1  [ 1, 2, 0, 0, 0]
row 2  [ 1, 2, 3, 0, 0]
row 3  [ 1, 2, 3, 4, 0]
row 4  [ 1, 2, 3, 4, 5]
```
And if we only focus on n = k condition:
```
n = k
         0  1  2  3  4
row 0  [ 1, 0, 0, 0, 0]
row 1  [ 0, 2, 0, 0, 0]
row 2  [ 0, 0, 3, 0, 0]
row 3  [ 0, 0, 0, 4, 0]
row 4  [ 0, 0, 0, 0, 5]
```
And finally we only focus on n > k condition:
```
n > k
         0  1  2  3  4
row 0  [ 1, 0, 0, 0, 0]
row 1  [ 1, 2, 0, 0, 0]
row 2  [ 1, 2, 3, 0, 0]
row 3  [ 1, 2, 3, 5, 0]
row 4  [ 1, 2, 3, 4, 5]
```
And if we put it all together we get:
```
         0  1  2  3  4
row 0  [ 1, 0, 0, 0, 0]
row 1  [ 1, 2, 0, 0, 0]
row 2  [ 1, 2, 3, 0, 0]
row 3  [ 1, 2, 3, 4, 0]
row 4  [ 1, 2, 3, 4, 5]
```

Now with this we have structured state space models (S4) but the matrices A and
B are the same for all time steps, all inputs. This means that if we try to
get the model to learn a specific pattern in the input sequence it will be
difficult as it does not take the input into consideration, it cannot do content
aware resoning (the parameters A, B, C and D are the same for all time steps).
So an SSM will treat all inputs equally.
But what does this really mean. Like my initial thought was that if I wanted the model
to selectively copy something from an input sequence then did not seen an
issue with doing that. My reasoning was that the tokens in the sequence are
processed one by one, so tokens earlier in the sequence would have been
integrated into the hidden state by the time the following sequence tokens are
processed, so in my mind it should/could be able to recall these. Now the issue
is not that the information is not there but it is more an issue of how
accessible/usable it is. As more tokens are processed the earlier tokens get
more and more compressed and mixed with newer tokens. And there is no direct
addressing like there is in the tranformers with the self-attention mechanism.
So the information is there but the S4 model lacks the ability to identify which
part of the hidden state corresponds to the asked for information and use that
in a targeted way for the current token.

The fixed nature of the model means it can't selectively choose which
information to retain or discard from its hidden state. It compresses all
information uniformly, potentially losing important distinctions between
different types of tokens

This is where _selective_ state space models come in. We want to selectively add
the data that will make up the state. To accopmlish this we need to parameters
that depend on the input. In the S4 model the matrices A, B, and C are static
and have a fixed size.  The SSM processes inputs sequentially, one step at a
time so even if B is multiplied by the input vector.

In S4 we have:
```
N = inner state size
D = size of input vector

   +-------+     +-------+     +-------+
   |  A    | N   |  B    | N   |  C    |  N
   |       |     |       |     |       |
   +-------+     +-------+     +-------+
      D             D             D

A: (D, N)
B: (D, N)
C: (N, D)
```
And in Mamba we have:
```
L = sequence length
D = size of input vector
N = inner state size
B = batch size

  \  B     \    \   B   \     \  B    \
   +-------+     +-------+     +-------+
   |  Δ    | L   |  B    | L   |  C    |  L
   |       |     |       |     |       |
   +-------+     +-------+     +-------+
      D             N             N

A: (D, N)
B: (B, L, N)
C: (B, L, N)
Δ: (B, L, N)
```
Notice that the above matrices are dynamic and depend on L, the input sequence
length. These matrices selectively incorporate information from the input into
the hidden state (and what do not include, that is what can be ignored).
And notice that this model can also handle batches of input sequences, like an
initial prompt.

In Mamba the B and C matrices are dynamic and depend on the input. Also the Δ
s dynamic and depends on the input. This means that the shapes of the matrices
also depend on the input.
```
Input: x : (B, L, D)
Output: y : (B, L, D)

1: A: (D, N)             Learned parameter
2: B: (B, L, N) S_B(x) = Linear_N(x)
3: C: (B, L, N) S_C(x) = Linear_N(x)
4: Δ: (B, L, N) S_Δ(x)
5: A, B: (B, L, D, N)  = descretize(Δ, A, B)
6: y = SSM(A, B, C,)(x)
```
Linear_N(x) is a linear transformation that projects each token of the input x
into the hidden space N. The input X has shape (B, L, D), and this projects it
to shape (B, L, N). The resulting B is specific to each input sequence, and
within that sequence, each token can have a different projection in the
N-dimensional space. The transformation itself (Linear_N) is the same for all
tokens and sequences, but its output varies based on the input content.

So the matrix B is different for each token in the input sequence. So B can be
different for each token in the sequence.
So for Linear_N(x) we would have a learned matrix W and learned vector b. So
for each token we would have:
```
W: (N, D)
b: (N)

y = Wx + b
And the whole sequence could be handled by:
Y = XW^T + b
And Y would have shape (B, L, N)
```
So the same weights and biases are used for all tokens but they transform each
token independently. So this is mapping the tokens to different points in the
N-dimensional space.
For C this is the same but a different matrix and vector.

For the delta (Δ) values they are generated by:
```
sΔ(𝑥) = Broadcast_D (Linear_1 (𝑥))

Linear_1 = is a projection into a single dimension (B, L, D) -> (B, L, 1)
Broadcast_D = takes the output of Linear_1 and it to D dimensions

1. Linear_1: (B, L, D) -> (B, L, 1)
             Weight: (1, D)
             Bias: (1)
             Operation: y = W^Tx + b
             Output: (B, L, 1)

2. Broadcast_D: (B, L, 1) -> (B, L, D)
                Operation: Takes the (B, L, 1) tensor and repeats it along the
                last dimension D times. So it just copies the same value into
                each element of the last dimension.
                Output: (B, L, D)
```
And Δ is then passed to the descretize function along with the A and B matrices.
Now, one thing that confused me a litle was that I read that A is a static
learned parameter in contrast to B and C which we can see are dynamically
generated by the functions above. But the descretize function takes A and
returns an updated/adapted version of it with sequence specific modifications.
This might look like:
```
A_discrete = exp(Δ * A)   (e^ΔA)

exp = matrix exponentiation
```
The discretize function creates input-specific state transition matrix each
forward pass. And note that this is not an element wise exponentiation but
a matrix exponentiation as this preserves the structure of the matrix (like
eigenvalues and eigenvectors).
So while A is a learned parameter it is adapted for each input token.

Now Δ is a dynamic parameter as we saw and this is called the scale factor or
delta. So recall that the original state space model is defined for a continous
time system.

### Continous State Space in Mamba
The continuous state space can be viewed simliar to an embedding space. So the
state space is a high-dimensional vector space where each dimension might
represent some abstract feature or concept. The state at any given point in time
is a point/vector in this space.

We might be able to think of the current state, the point/vector, as the last
token's position in this space. And if the next token is "simliar" perhaps
semantically we might only transform/move the state a little, but if they are
not very simliar we might move the point further.
For example, if the we processed the token representing "cat" followed by a
token representing "kitten" we might only move a little as the context is not
changing very much. But if there is a token representing "microbe" we might move
further as this is a different context.

We can think of the current state as a point/vector representing the model's
understanding up to and including the last processed token.

So the current state of the system is a vector that has a point somewhere in
this space. The A matrix is a transformation that moves this vector to a new
point in the space. It suggests a direction the this point would naturally drift.
The B matrix determines how new input tokens influence the states position.

This is where delta comes in where it can control how much the state is
transformed.

Now, we need to keep in mind that we are dealing with an underlying continious
system, and A represents this continious-time dynamics. This might be described
a a differential equation like:
```
  dx
  -- = A x(t)
  dt

x(t) = state at time t
A    = state transition matrix
```
This describes how the state evolves over time without any input. So A is this
State transition matrix which describes how the system would evolve natrually
over time if there was not input to the system.
In Mamba we have a discrete system so we are approximating this continous system
with a discrete one:
```
x[t+1] = exp(Δ * A) * x[t]

x(t) = state at time t
A    = state transition matrix
Δ    = time step
```
Notice that there is still no input here.
The B matrix allows the input to influence the state transition.
```
Δ * B * u[t]

u[k] = input at time t
Δ    = time step
B    = input matrix
```
The `exp(Δ * A) * x[t]` part evolves the current state as if no input occurred.
The `Δ * B * u[t]` part adds the effect of the current input.
The combination of these two parts is the new state after processing the
current token:
```
x[t+1] = exp(Δ * A) * x[t] + Δ * B * u[t]
```
And again the delta is dynamic and can be adjusted for each token and can
control the step size (smaller steps to stay within the current context, larger
to perhaps move to a different context).


And lastly we have the SSM function has the parameters A, B, and C matrices and
takes as input the sequence of tokens:

```
y = SSM(A, B, C,)(x)
```
Now, even though SSM takes the whole sequence of input tokens (B, L, D) it
operates on each token independently. 

_wip_


We can visualize this as
```
    +---+      +---+       +---+
----| x |------| h |---+---| y |----------------------------->
    +---+  ↑   +---+   |   +---+
           |   +---+   |
           +---| A |---+
               +---+
```

### Convolution layer
In the block diagram above we first have a projection followed by a convolution:
```
        input
          |
          |-----------------------------+
          ↓                             |
     +-----------+                +------------+
    / projection  \              /  projection  \
   +---------------+            +----------------+
          |                            |
          ↓                            |
   +---------------+                   |
   | convolution   |                   |
   +---------------+                   |
          |                            |
          ↓                            ↓
```
So the input consists of a sequence of tokens embeddings. The projection layer
simply performs a linear transformation of the input embeddings to a higher
dimensions. This higher dimension is specified as `d_inner`.

Next we have the convolution layer. Now if we recall how a SSM works we have
the internal state which captures information about past tokens. But we also
want to be able to take the current tokens into account and their interactions
with each other. So the convoluation is about capturing local interactions
efficiently.
So we start with the token embeddings, and then we project them to a higher
dimension and it is this higher dimension that the convolution is applied to.

```
Input sequence: "Dan loves icecream", That might be tokenized in to
Tokens        : [2223, 25, 883, 10033]
embeddings    :
                 2223 : [1 2 3 4 5 6 7 8]
                   25 : [9 8 7 6 5 4 3 2]
                  883 : [1 2 3 4 5 6 7 8]
                10033 : [9 8 7 6 5 4 3 2]
```
Now, the projection will take the embeddings vectors/matrix as input and
increase the dimensions of these vectors. This is done using a learned matrix.
The size of this matrix would be (embedding_size, projection_size). So if we
wanted the increase the dimensions by 8 we would use (8, 16).
```
                 2223 : [1 2 3 4 5 6 7 8 ... 15 16]
                   25 : [9 8 7 6 5 4 3 2 ... 15 16]
                  883 : [1 2 3 4 5 6 7 8 ... 15 16]
                10033 : [9 8 7 6 5 4 3 2 ... 15 16]
```
This would then be the input to the convolution operation. Now, remember that
the convolution is about capturing local interactions. So we have a kernel
matrix that is applied to the input. The kernel matrix is a learned matrix. So
will have a matrix of size (kernel_size, projection_size):
```
(kernel_size, projection_size) = (3, 16)
(    3         ,           16)
       
        0       1     2
 0   [w_0_0  w_0_1  w_0_1 ]
     [w_1_2  w_2_2  w_3_2 ]
     [w_1_3  w_2_3  w_3_3 ]
     [ ...    ...   ...   ]
15   [w_1_16 w_2_16 w_3_16]
```
This filter will be applied to each projected token embeddings the number of
elements in the token embeddings equal to the kernel size, 3 in our case. 

```
Step 1 (first 3 positions):

 Input           Kernel           Output
 [1 2 3]   [w_0_0  w_0_1  w_0_2 ]   [y_0] 
 [9 8 7]   [w_1_0  w_1_1  w_1_2 ]   [y_1]
 [1 2 3]   [w_2_0  w_2_1  w_2_2 ]   [y_2]
 [9 8 7]   [w_3_0  w_3_1  w_3_2 ]   [y_3]
           [w_4_0  w_4_1  w_4_2 ]   [y_4]
           [w_5_0  w_5_1  w_5_2 ]   [y_5]
           [w_6_0  w_6_1  w_6_2 ]   [y_6]
           [w_7_0  w_7_1  w_7_2 ]   [y_7]
           [w_8_0  w_8_1  w_8_2 ]   [y_8]
           [w_9_0  w_9_1  w_9_2 ]   [y_9]
           [w_10_0 w_10_1 w_10_2]   [y_10]
           [w_11_0 w_11_1 w_11_2]   [y_11]
           [w_12_0 w_12_1 w_12_2]   [y_12]
           [w_13_0 w_13_1 w_13_2]   [y_13]
           [w_14_0 w_14_1 w_14_2]   [y_14]
           [w_15_0 w_15_1 w_15_2]   [y_15]

y_i = Σ(Input_j[0:3] ⊙ Kernel[i][0:3]) for j = 1 to 4

So for y0:
y_0 = ((1 * w_0_0) + (2 * w_0_1) + (3 * w_0_2)) +
      ((9 * w_0_0) + (8 * w_0_1) + (7 * w_0_2)) +
      ((1 * w_0_0) + (2 * w_0_1) + (3 * w_0_2)) +
      ((9 * w_0_0) + (8 * w_0_1) + (7 * w_0_2))

Step 2 (kernel slides one positions to the right):

 Input           Kernel           Output
 [2 3 4]   [w_0_0  w_0_1  w_0_2 ]   [y_0] 
 [8 7 6]   [w_1_0  w_1_1  w_1_2 ]   [y_1]
 [2 3 4]   [w_2_0  w_2_1  w_2_2 ]   [y_2]
 [8 7 6]   [w_3_0  w_3_1  w_3_2 ]   [y_3]
           [w_4_0  w_4_1  w_4_2 ]   [y_4]
           [w_5_0  w_5_1  w_5_2 ]   [y_5]
           [w_6_0  w_6_1  w_6_2 ]   [y_6]
           [w_7_0  w_7_1  w_7_2 ]   [y_7]
           [w_8_0  w_8_1  w_8_2 ]   [y_8]
           [w_9_0  w_9_1  w_9_2 ]   [y_9]
           [w_10_0 w_10_1 w_10_2]   [y_10]
           [w_11_0 w_11_1 w_11_2]   [y_11]
           [w_12_0 w_12_1 w_12_2]   [y_12]
           [w_13_0 w_13_1 w_13_2]   [y_13]
           [w_14_0 w_14_1 w_14_2]   [y_14]
           [w_15_0 w_15_1 w_15_2]   [y_15]

The process continues until the kernel has been applied to all the tokens in the
input sequence.
```
So what we have done here is that we have taken three features from from the
projected input embeddings and applied the kernel to them. This gives as a
weighed sum of these features for accross all the tokens (mixing them together
in a sense). This is what enables the capturing of local neighbors information.
And notice that the kernel slides one position to the right at a time which
means the length of the output will be the same as the input.

#### Padding
Now, we also need to consider passing to ensure that the output length is
correct. What I means is that consider the following input sequence of 4 tokens:
```
Input: [A, B, C, D]
Kernal size: 3

Step 1: [A B C] -> y_0
Step 2: [B C D] -> y_1
```
There are no more steps to take and the output length has gone from 4 to 2. We
can fix this by adding padding to the beginning of the input (for causual
models):
```
Input: [0, 0, A, B, C, D]
Kernal size: 3
Step 1: [0 0 A] -> y_0
Step 2: [0 A B] -> y_1
Step 3: [A B C] -> y_2
Step 4: [B C D] -> y_3
```
How do we know how much padding to add? Well, the padding size is equal to the
kernel size minus 1. We need d_conv total elements for the first computation
plus the "real" input value for the first step. This is the reason for the minus
one.

To better understand how the convolution works there is a standalone example
in [ssm_conv.c](../../fundamentals/ggml/src/ssm_conv.c).

### Mamba in llama.cpp
This section will take a look at how Mamba in implemented in llama.cpp.

#### Preperation
Download a Mamba model that we can use for the example:
```
$ cd fundamentals/llama.cpp
$ make donwload-mamba
```
Can compile the example we will be using:
```console
$ make simple-prompt-multi
```

#### Inspect the Mamba model
```console
$ make inspect-mamba-model
INFO:gguf-dump:* Loading: models/mamba-1.4b-f16.gguf
* File is LITTLE endian, script is running on a LITTLE endian host.
* Dumping 25 key/value pair(s)
      1: UINT32     |        1 | GGUF.version = 3
      2: UINT64     |        1 | GGUF.tensor_count = 482
      3: UINT64     |        1 | GGUF.kv_count = 22
      4: STRING     |        1 | general.architecture = 'mamba'
      5: STRING     |        1 | general.name = 'mamba-1.4b-hf'
      6: UINT32     |        1 | mamba.context_length = 1048576
      7: UINT32     |        1 | mamba.embedding_length = 2048
      8: UINT32     |        1 | mamba.feed_forward_length = 0
      9: UINT32     |        1 | mamba.attention.head_count = 0
     10: UINT32     |        1 | mamba.block_count = 48
     11: UINT32     |        1 | mamba.ssm.conv_kernel = 4
     12: UINT32     |        1 | mamba.ssm.inner_size = 4096
     13: UINT32     |        1 | mamba.ssm.state_size = 16
     14: UINT32     |        1 | mamba.ssm.time_step_rank = 128
     15: FLOAT32    |        1 | mamba.attention.layer_norm_rms_epsilon = 9.999999747378752e-06
     16: UINT32     |        1 | general.file_type = 1
     17: STRING     |        1 | tokenizer.ggml.model = 'gpt2'
     18: [STRING]   |    50280 | tokenizer.ggml.tokens
     19: [INT32]    |    50280 | tokenizer.ggml.token_type
     20: [STRING]   |    50009 | tokenizer.ggml.merges
     21: UINT32     |        1 | tokenizer.ggml.bos_token_id = 0
     22: UINT32     |        1 | tokenizer.ggml.eos_token_id = 0
     23: UINT32     |        1 | tokenizer.ggml.unknown_token_id = 0
     24: UINT32     |        1 | tokenizer.ggml.padding_token_id = 0
     25: UINT32     |        1 | general.quantization_version = 2
* Dumping 482 tensor(s)
      1:  102973440 |  2048, 50280,     1,     1 | F16     | token_embd.weight
      2:      65536 |    16,  4096,     1,     1 | F32     | blk.0.ssm_a
      3:       4096 |  4096,     1,     1,     1 | F32     | blk.0.ssm_d
      4:       4096 |  4096,     1,     1,     1 | F32     | blk.0.ssm_conv1d.bias
      5:      16384 |     4,  4096,     1,     1 | F32     | blk.0.ssm_conv1d.weight
      6:       4096 |  4096,     1,     1,     1 | F32     | blk.0.ssm_dt.bias
      7:     524288 |   128,  4096,     1,     1 | F32     | blk.0.ssm_dt.weight
      8:   16777216 |  2048,  8192,     1,     1 | F16     | blk.0.ssm_in.weight
      9:    8388608 |  4096,  2048,     1,     1 | F16     | blk.0.ssm_out.weight
     10:     655360 |  4096,   160,     1,     1 | F32     | blk.0.ssm_x.weight
     11:       2048 |  2048,     1,     1,     1 | F32     | blk.0.attn_norm.weight

     12:      65536 |    16,  4096,     1,     1 | F32     | blk.1.ssm_a
     13:       4096 |  4096,     1,     1,     1 | F32     | blk.1.ssm_d
     14:       4096 |  4096,     1,     1,     1 | F32     | blk.1.ssm_conv1d.bias
     15:      16384 |     4,  4096,     1,     1 | F32     | blk.1.ssm_conv1d.weight
     16:       4096 |  4096,     1,     1,     1 | F32     | blk.1.ssm_dt.bias
     17:     524288 |   128,  4096,     1,     1 | F32     | blk.1.ssm_dt.weight
     18:   16777216 |  2048,  8192,     1,     1 | F16     | blk.1.ssm_in.weight
     19:    8388608 |  4096,  2048,     1,     1 | F16     | blk.1.ssm_out.weight
     20:     655360 |  4096,   160,     1,     1 | F32     | blk.1.ssm_x.weight
     21:       2048 |  2048,     1,     1,     1 | F32     | blk.1.attn_norm.weight
     ...
    482:       2048 |  2048,     1,     1,     1 | F32     | output_norm.weight
```

### Overview of forward pass
```
        input
          |
          ↓
   +---------------+
   |     Norm      |
   +---------------+
          |
          |-----------------------------+
          ↓                             |
     +-----------+                +------------+
    / projection  \              /  projection  \
   +---------------+            +----------------+
          |                            |
          ↓                            |
   +---------------+                   |
   | convolution   |                   |
   +---------------+                   |
          |                            |
          ↓                            ↓
   +---------------+            +----------------+
   |     Silu      |            |    Silu        |
   +---------------+            +----------------+
          |                            |
          ↓                            |
   +---------------+                   |
   |     SSM       |                   |
   +---------------+                   |
          |                            |
          ↓                            |
   +---------------+                   |
   | activation    |←------------------+
   +---------------+
          |
          ↓
   +---------------+
    \  projection /
     +-----------+
          |
          ↓
       output
```

The input embedding in Mamba goes through a projection from the input embedding
space to the state space. The input tokens will have an embedding that the  
model uses, for example 2048. The input vector will go through a linear project
to the inner space dimension (`d_inner`)
That is the input projection, next comes the convolution.

### Selective scan 
The basic SSM has the following equation:
```
h[t] = Ah[t-1] + Bx[t]
y[t] = Ch[t]
```
In Mamba this is modified to take into account the current tokens:
```
h[t] = (A - exp(-Δ[t])) * h[t-1] + Bx[t]
y[t] = Ch[t]

A    = the learned state transition matrix.
Δ[t] = is an input dependant step size (one for each time step).
x[t] = input at time step t.
B    = input matrix (input dependent)
C    = output matrix (input dependent)
```


#### Exploration
```console
$ make debug-mamba-simple-prompt
gdb --args ./simple-promt 0 0 models/mamba-1.4b-f16.gguf
(gdb) br build_mamba
```

This is a very basic example and will create a batch with one sequence and it
will contains the tokens for the current prompt:
```c++
    std::string prompt = "What is LoRA?";
```
Note that I started this example using `simple-prompt-multi` which uses to
sequences but I later changed this back as I think at least for the first
iteration through the code it will be easer to understand with just one
sequence. But the multi sequence version will be used when we want to understand
the usage of the kv-cache, similar to what we did with RWKV. But there might
be inaccurate comments below with code/debugging output that is not correct but
I'll hopefully fix them all eventually.

```c++
    struct ggml_cgraph * build_mamba() {
        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, llama_model_max_nodes(model), false);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        // {n_embd, n_tokens}
        inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

        struct ggml_tensor * state_copy = build_inp_s_copy();
        struct ggml_tensor * state_mask = build_inp_s_mask();
        for (int il = 0; il < n_layer; ++il) {
           ...
```

```console
(gdb) p batch
$1 = (const llama_ubatch &) @0x7fffffffd4f0:
{equal_seqs = true, n_tokens = 5, n_seq_tokens = 5, n_seqs = 1,
token = 0x555555ac8610, embd = 0x0, pos = 0x555555ac8630,
n_seq_id = 0x555555ac8650, seq_id = 0x555555ac7ce0, output = 0x555555ac8670 ""}
```
So if we inspect `inpL` we see that shape is:
```console
(gdb) p *inpL
$2 = {type = GGML_TYPE_F32, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x0,
ne = {2048, 5, 1, 1}, nb = {4, 8192, 40960, 40960},
op = GGML_OP_GET_ROWS, op_params = {0 <repeats 16 times>}, flags = 0, grad = 0x0, src = {0x555556726700, 0x7fff4ae40980, 0x0, 0x0,
0x0, 0x0, 0x0, 0x0, 0x0, 0x0}, view_src = 0x0, view_offs = 0, data = 0x0,
name = "inp_embd", '\000' <repeats 55 times>,
extra = 0x0}
```
So the `inpL` tensor is storing 5 token embeddings representing our prompt.

We can inspect the embedding size using:
```console
(gdb) p lctx.model.hparams.n_embd
$75 = 2048
```
Notice that similar to RWKV we are creating a `state_copy` tensor and a 
`state_mask` tensor. The model in this case has 48 layers so we will iterate
over them.

For each layer we will have a normalization.
```c++
            // norm
            cur = llm_build_norm(ctx0, inpL, hparams,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, cb, il);
            cb(cur, "attn_norm", il);

            cur = llm_build_mamba(ctx0, lctx, batch, gf, cur,
                    state_copy, state_mask,
                    kv_head, n_kv, cb, il);
```
And then we have the Mamba layer:
```c++
static struct ggml_tensor * llm_build_mamba(
        struct ggml_context * ctx,
       struct llama_context & lctx,
         const llama_ubatch & batch,
         struct ggml_cgraph * graph,
         struct ggml_tensor * cur,
         struct ggml_tensor * state_copy,
         struct ggml_tensor * state_mask,
                    int32_t   kv_head,
                    int32_t   n_kv,
         const llm_build_cb & cb,
                    int       il) {
    const llama_model    & model   = lctx.model;
    const llama_hparams  & hparams = model.hparams;
    const llama_kv_cache & kv      = lctx.kv_self;
    const int64_t d_conv  = hparams.ssm_d_conv;
    const int64_t d_inner = hparams.ssm_d_inner;
    const int64_t d_state = hparams.ssm_d_state;
    const int64_t dt_rank = hparams.ssm_dt_rank;
    const int64_t n_seqs  = batch.n_seqs;
```
Just a note about `model.hparams`. When a model is loaded by `llama_model_load`
that function will call:
```c++
        try {
            llm_load_hparams(ml, model);
        } catch(const std::exception & e) {
            throw std::runtime_error("error loading model hyperparameters: " + std::string(e.what()));
        }
```
The `llm_load_hparams` function will load the hyperparameters from the model
passed in. This is how `hparams` is populated. `llm_load_hparams` has a switch
statement and a case for Mamba:
```
        case LLM_ARCH_MAMBA:
            {
                ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
                ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
                ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
                ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
                ml.get_key(LLM_KV_SSM_DT_B_C_RMS, hparams.ssm_dt_b_c_rms, false);

                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                switch (hparams.n_layer) {
                    case 24:
                        switch (hparams.n_embd) {
                            case 768: model.type = e_model::MODEL_SMALL; break;
                            default: model.type = e_model::MODEL_UNKNOWN;
                        } break;
                    case 48:
                        switch (hparams.n_embd) {
                            case 1024: model.type = e_model::MODEL_MEDIUM; break;
                            case 1536: model.type = e_model::MODEL_LARGE; break;
                            case 2048: model.type = e_model::MODEL_XL; break;
                            default: model.type = e_model::MODEL_UNKNOWN;
                        } break;
                    case 64:
                        switch (hparams.n_embd) {
                            case 2560: model.type = e_model::MODEL_3B; break;
                            default: model.type = e_model::MODEL_UNKNOWN;
                        } break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
```
These are the hyperparameters specific to the Mamba model:
```c++
struct llama_hparams {
    // for State Space Models
    uint32_t ssm_d_conv  = 0;
    uint32_t ssm_d_inner = 0;
    uint32_t ssm_d_state = 0;
    uint32_t ssm_dt_rank = 0;
    bool ssm_dt_b_c_rms = false;
```

So back to `llm_build_mamba`:
```c++
static struct ggml_tensor * llm_build_mamba(
        struct ggml_context * ctx,
       struct llama_context & lctx,
         const llama_ubatch & batch,
         struct ggml_cgraph * graph,
         struct ggml_tensor * cur,
         struct ggml_tensor * state_copy,
         struct ggml_tensor * state_mask,
                    int32_t   kv_head,
                    int32_t   n_kv,
         const llm_build_cb & cb,
                    int       il) {
    const llama_model    & model   = lctx.model;
    const llama_hparams  & hparams = model.hparams;
    const llama_kv_cache & kv      = lctx.kv_self;
    const int64_t d_conv  = hparams.ssm_d_conv;
    const int64_t d_inner = hparams.ssm_d_inner;
    const int64_t d_state = hparams.ssm_d_state;
    const int64_t dt_rank = hparams.ssm_dt_rank;
    const int64_t n_seqs  = batch.n_seqs;
```
Lets inspect these variables:
```console
(gdb) p d_conv
$21 = 4
(gdb) p d_inner
$22 = 4096
(gdb) p d_state
$23 = 16
```
`d_conv` is the size of the kernel that will be used in the convolution.
`d_inner` is the size of the inner space, what the projection layer projected
the input embeddings to (from the input embedding dimension
`lctx.model.hparams.n_embd model.n`).

Next we have:
```c++
    struct ggml_tensor * conv_states_all = kv.k_l[il];
    struct ggml_tensor * ssm_states_all  = kv.v_l[il];

    // (ab)using the KV cache to store the states
    struct ggml_tensor * conv = llm_build_copy_mask_state(ctx,
            graph, conv_states_all, state_copy, state_mask,
            hparams.n_embd_k_s(), kv.size, kv_head, n_kv, n_seqs);
```
Again, simliar to what we went through with RWKV we are calling
`llm_build_copy_mask_state` and passing in `kv.k_l[i]`. This will return a
tensor with the following size:
```console
(gdb) p *conv
gdb) p *conv
$10 = {type = GGML_TYPE_F32, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x0,
ne = {12288, 1, 1, 1}, nb = {4, 49152, 49152, 49152},
op = GGML_OP_VIEW, op_params = {0 <repeats 16 times>}, flags = 0, grad = 0x0, src = {0x7fff4ae41500, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
0x0, 0x0, 0x0}, view_src = 0x7fff4ae41500, view_offs = 0, data = 0x0, name = "node_2 (view)", '\000' <repeats 50 times>,
extra = 0x0}
```
This will then get reshaped from a 2d tensor to a 3d tensor:
```c++
    conv = ggml_reshape_3d(ctx, conv, d_conv - 1, d_inner, n_seqs);
```
Notice that this making the x dimension `d_conv - 1` which is something we also
did in the padding section above. That will result in a tensor like this:
```console
(gdb) p *conv
$11 = {type = GGML_TYPE_F32, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x0,
ne = {3, 4096, 1, 1}, nb = {4, 12, 49152, 49152}, op = GGML_OP_RESHAPE,
op_params = {0 <repeats 16 times>}, flags = 0, grad = 0x0, src = {0x7fff4ae41ac0, 0x0, 0x0, 0x0, 0x0, 0x0,
0x0, 0x0, 0x0, 0x0}, view_src = 0x7fff4ae41500, view_offs = 0, data = 0x0,
name = "node_2 (view) (reshaped)", '\000' <repeats 39 times>, extra = 0x0}
```
And we can visualize the sequence:
```
 0   0       2
     [       ]
        ...
        ...
        ...
        ...
        ...
4095 [       ]
```
I'll get back to the `conv` tensor shortly and I'd also like to address what
this is storing in the cache as that is not clear to me yet.
TOOD: revisit and explain what is being cached.

Next we do setup the `ssm` tensor by calling `llm_build_copy_mask_state` and the
reshape it:
```c++
    struct ggml_tensor * ssm = llm_build_copy_mask_state(ctx,
            graph, ssm_states_all, state_copy, state_mask,
            hparams.n_embd_v_s(), kv.size, kv_head, n_kv, n_seqs);
```
But this tensor will have a different shape:
```console
(gdb) p *ssm
$12 = {type = GGML_TYPE_F32, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x0,
ne = {65536, 1, 1, 1}, nb = {4, 262144, 262144, 262144}, op = GGML_OP_VIEW,
op_params = {0 <repeats 16 times>}, flags = 0, grad = 0x0,
src = {0x7fff4ae42080, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0},
view_src = 0x7fff4ae42080, view_offs = 0, data = 0x0, name = "node_8 (view)", '\000' <repeats 50 times>,
extra = 0x0}
```
And this will be reshaped into 16x4096x1:
```c++
    ssm = ggml_reshape_3d(ctx, ssm, d_state, d_inner, n_seqs);
```
```console
(gdb) p *ssm
$13 = {type = GGML_TYPE_F32, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x0,
ne = {16, 4096, 1, 1}, nb = {4, 64, 262144, 262144}, op = GGML_OP_RESHAPE,
op_params = {0 <repeats 16 times>}, flags = 0, grad = 0x0, src = {0x7fff4ae42640, 0x0, 0x0, 0x0, 0x0, 0x0,
0x0, 0x0, 0x0, 0x0}, view_src = 0x7fff4ae42080, view_offs = 0, data = 0x0,
name = "node_8 (view) (reshaped)", '\000' <repeats 39 times>, extra = 0x0}
```

After that `cur` will be reshaped from 2048x5 to 2048x5x1:
```c++
    cur = ggml_reshape_3d(ctx, cur, cur->ne[0], n_seq_tokens, n_seqs);
```
```console
(gdb) p *cur
$17 = {type = GGML_TYPE_F32, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x0,
ne = {2048, 5, 1, 1}, nb = {4, 8192, 40960, 40960}, op = GGML_OP_RESHAPE, op_params = {0 <repeats 16 times>}, flags = 0, grad = 0x0, src = {0x7fff4ae410b0, 0x0, 0x0, 0x0, 0x0, 0x0,
0x0, 0x0, 0x0, 0x0}, view_src = 0x7fff4ae410b0, view_offs = 0, data = 0x0,
name = "attn_norm-0 (reshaped)", '\000' <repeats 41 times>, extra = 0x0}
```

Next we are going to multiply the reshaped `cur` with `model.layers[i].ssm_in`
```c++
    struct ggml_tensor * xz = llm_build_lora_mm(lctx, ctx, model.layers[il].ssm_in, cur);
```
Is this the projection perhaps?
```console
(gdb) p *xz
$18 = {type = GGML_TYPE_F32, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x0,
ne = {8192, 5, 1, 1}, nb = {4, 32768, 163840, 163840}, op = GGML_OP_MUL_MAT,
op_params = {0 <repeats 16 times>}, flags = 0, grad = 0x0,
src = {0x555556726cc0, 0x7fff4ae42920, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0},
view_src = 0x0, view_offs = 0, data = 0x0, name = '\000' <repeats 63 times>, extra = 0x0}
```
I think this is the projection. Next a view of xz:
```c++
    //                                             4096        5         1        32768      163840
    struct ggml_tensor * x = ggml_view_3d(ctx, xz, d_inner, xz->ne[1], xz->ne[2], xz->nb[1], xz->nb[2], 0);
```
This is the first half of sx. And then we take the second have by specyfing the
same sizes but changing the offset to be `d_inner*ggml_element_size(xz)`:
```c++
    struct ggml_tensor * z = ggml_view_3d(ctx, xz, d_inner, xz->ne[1], xz->ne[2], xz->nb[1], xz->nb[2], d_inner*ggml_element_size(xz));
```
So at this point we are at the projection layer of a Mamba block (refering to
the diagram we have above).

Next, we have the convolution:
```c++
    // conv
    {
        // => {d_conv - 1 + n_seq_tokens, d_inner, n_seqs}
        struct ggml_tensor * conv_x = ggml_concat(ctx, conv, ggml_transpose(ctx, x), 0);
```
We can inspect the transposed x tensor:
```console
(gdb) p *ggml_transpose(ctx, x)
$25 = {type = GGML_TYPE_F32, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x0,
ne = {5, 4096, 1, 1}, nb = {32768, 4, 163840, 163840}, op = GGML_OP_TRANSPOSE,
op_params = {0 <repeats 16 times>}, flags = 0, grad = 0x0, src = {0x7fff4ae42c00, 0x0, 0x0, 0x0, 0x0, 0x0,
0x0, 0x0, 0x0, 0x0}, view_src = 0x7fff4ae42a90, view_offs = 0, data = 0x0,
name = " (view) (transposed)", '\000' <repeats 43 times>, extra = 0x0}
```
And we are using concat and specifying the first dimension:
```console
(gdb) p *conv_x
$27 = {type = GGML_TYPE_F32, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x0,
ne = {8, 4096, 1, 1}, nb = {4, 32, 131072, 131072}, op = GGML_OP_CONCAT,
op_params = {0 <repeats 16 times>}, flags = 0, grad = 0x0, src = {0x7fff4ae41c30, 0x7fff4ae43050, 0x0, 0x0,
0x0, 0x0, 0x0, 0x0, 0x0, 0x0}, view_src = 0x0, view_offs = 0, data = 0x0, name = '\000' <repeats 63 times>, extra = 0x0}
```
Notice that we now have a size of 8 elements for the first dimension instead of
3. So `conv` is the convolution which before this operation has a size of 3 for
its first dimension (x dimension). But our input length is 5.
```
Input: [A B C D E]
Kernal size: 3
Step 1: [A B C D] -> y_0
Step 1: [B C D E] -> y_1
```
So that would only produce 2 outputs. We can fix this by adding padding to the
beginning of the input just like we did in the padding section above::
```
Input: [0 0 0 A B C D E]
Step 1: [0 0 0 A] -> y_0
Step 2: [0 0 A B] -> y_1
Step 3: [0 A B C] -> y_2
Step 4: [A B C D] -> y_3
Step 5: [B C D E] -> y_4
```

TOOD: Revisit the following when using multiple sequences.
```c++
        // copy last (d_conv - 1) columns back into the state cache
        struct ggml_tensor * last_conv = ggml_view_3d(ctx, conv_x, d_conv - 1, d_inner, n_seqs, conv_x->nb[1], conv_x->nb[2], n_seq_tokens*(conv_x->nb[0]));

        ggml_build_forward_expand(graph,
            ggml_cpy(ctx, last_conv,
                ggml_view_1d(ctx, conv_states_all,
                    (d_conv - 1)*(d_inner)*(n_seqs),
                    kv_head*(d_conv - 1)*(d_inner)*ggml_element_size(conv_states_all))));
```

Next we have the convolution operation creation (recall that we are only
building up the computation graph at this stage and the actual operation will
happen during decoding later):

So the tensor that we want to apply the convolution to is `conv_x` and the
kernel is defined by `model.layers[il].ssm_conv1d`:
```c++
    x = ggml_ssm_conv(ctx, conv_x, model.layers[il].ssm_conv1d);
```
And the kernel has the following shape:
```console
(gdb) p *model.layers[il].ssm_conv1d
$32 = {type = GGML_TYPE_F32, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x555556717cb0,
ne = {4, 4096, 1, 1}, nb = {4, 16, 65536, 65536}, op = GGML_OP_NONE,
op_params = {0 <repeats 16 times>}, flags = 0, grad = 0x0, src = {0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
0x0, 0x0, 0x0}, view_src = 0x0, view_offs = 0, data = 0x7fff58666420,
name = "blk.0.ssm_conv1d.weight", '\000' <repeats 40 times>, extra = 0x0}
```
Then a bias is added:
```c++
        x = ggml_add(ctx, x, model.layers[il].ssm_conv1d_b);
```
And then we have the Silu:
```c++
        x = ggml_silu(ctx, x);
```
And that is the end of the convolution block.

Next we have the SSM block in the diagram (and code).
The output from the convolution is the input to this block and this is in the
shape of:
```console
(gdb) p *x
$61 = {type = GGML_TYPE_F32, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x0,
ne = {4096, 5, 1, 1}, nb = {4, 16384, 81920, 81920},
op = GGML_OP_UNARY, op_params = {10, 0 <repeats 15 times>}, flags = 0,
grad = 0x0, src = {0x7fff4ae43780, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0},
view_src = 0x0, view_offs = 0, data = 0x0, name = '\000' <repeats 63 times>, extra = 0x0}
```
So first this will be projected by using `model.layers[il].ssm_x`:
```c++
    // ssm
    {
        // {d_inner, dt_rank + 2*d_state} @ {d_inner, n_seq_tokens, n_seqs} => {dt_rank + 2*d_state, n_seq_tokens, n_seqs}
        struct ggml_tensor * x_db = llm_build_lora_mm(lctx, ctx, model.layers[il].ssm_x, x);
```
This will actually reduce the size compared to the x tensor:
```console
(gdb) p *x_db
$66 = {type = GGML_TYPE_F32, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x0,
ne = {160, 5, 1, 1}, nb = {4, 640, 3200, 3200}, op = GGML_OP_MUL_MAT,
op_params = {0 <repeats 16 times>}, flags = 0, grad = 0x0, src = {0x555556727110, 0x7fff4ae438f0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0}, view_src = 0x0, view_offs = 0, data = 0x0, name = '\000' <repeats 63 times>, extra = 0x0}
```
I wonder what the reason for this reduction is?

Following that we will create a view for the delta tensor:
```c++
        struct ggml_tensor * dt = ggml_view_3d(ctx, x_db, dt_rank, n_seq_tokens, n_seqs, x_db->nb[1], x_db->nb[2], 0);
```

```console
(gdb) p dt_rank
$67 = 128
(gdb) p *dt
$72 = {type = GGML_TYPE_F32, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x0,
ne = {128, 5, 1, 1}, nb = {4, 640, 3200, 3200}, op = GGML_OP_VIEW,
op_params = {0 <repeats 16 times>}, flags = 0, grad = 0x0, src = {0x7fff4ae43a60, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
0x0, 0x0, 0x0}, view_src = 0x7fff4ae43a60, view_offs = 0, data = 0x0, name = " (view)", '\000' <repeats 56 times>, extra = 0x0}
```
And the we create a view for the B tensor:
```c++
        struct ggml_tensor * B  = ggml_view_3d(ctx, x_db, d_state, n_seq_tokens, n_seqs, x_db->nb[1], x_db->nb[2], ggml_element_size(x_db)*dt_rank);
```
```console
(gdb) p *B
$74 = {type = GGML_TYPE_F32, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x0,
ne = {16, 5, 1, 1}, nb = {4, 640, 3200, 3200}, op = GGML_OP_VIEW,
op_params = {512, 0 <repeats 15 times>}, flags = 0, grad = 0x0, src = {0x7fff4ae43a60, 0x0, 0x0, 0x0, 0x0, 0x0,
0x0, 0x0, 0x0, 0x0}, view_src = 0x7fff4ae43a60, view_offs = 512, data = 0x0, name = " (view)", '\000' <repeats 56 times>,
extra = 0x0}
```
And then C:
```c++
        struct ggml_tensor * C  = ggml_view_3d(ctx, x_db, d_state, n_seq_tokens, n_seqs, x_db->nb[1], x_db->nb[2], ggml_element_size(x_db)*(dt_rank+d_state));
```
```console
(gdb) p *C
$75 = {type = GGML_TYPE_F32, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x0,
ne = {16, 5, 1, 1}, nb = {4, 640, 3200, 3200}, op = GGML_OP_VIEW,
op_params = {576, 0 <repeats 15 times>}, flags = 0, grad = 0x0,
src = {0x7fff4ae43a60, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0}, view_src = 0x7fff4ae43a60, view_offs = 576, data = 0x0, name = " (view)", '\000' <repeats 56 times>,
extra = 0x0}
```
(wip)
