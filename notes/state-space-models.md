## State Space Models (SSM)
State of a system is the set of variables which at some point in time, together
with the inputs to the system, completely determines the behaviour of the
system.

SSM is a broad class of models that can be used to describe the behaviour of a
system over time and used in various domains such as control theory, robotics,
and machine learning.

#### Structured State Space Sequence Models (S4)
Is a type of SSM that is speciallized for the domain of deep learning and
sequential modeling.

Simliar to RNNs, S4 models are designed to process sequential data specifically
where the order of input matters (NLP, audio signal processing).
And the S4 would be used in place of a convolutional layer in a CNN, or the
attention mechanism in a transformer. So if we have a LLM we would have have
the SSM where we would see the attention layer(s) in a Transformer
architectures. These are sometimes called Deep SSMs.

Both architectures maintain a form of state across time steps. In RNNs, this is
the hidden state, while in S4 models, it's the state vector `x`.

In RNNs, the parameters of the model are the weights and biases of the network,
whereas in S4 models, the parameters are the state transition matrix `A`, the
input matrix `B`, the output matrix `C`, and the feedthrough matrix `D`.

The formula for S4 models is:
```
hₜ₊₁ = Axₜ + Bxₜ
yₜ = Cxₜ + Duₜ

hₜ₊₁ = is the state at time t+1
A    = state transition matrix
xₜ   = is the input at time t
B    = input projection matrix
C    = output matrix
D    = feedthrough matrix
```
We can visualize this as
```
    +---+      +---+       +---+
----| U |------| X |---+---| y |----------------------------->
    +---+  ↑   +---+   |   +---+
           |   +---+   |
           +---| A |---+
               +---+
```

Lets take an example, the inputs we have are tokens but well represent them
as [u₁, u₂, u₃] and they could for example represent words in a sentence:
```
A = 0.5
B = 1
C = 2
D = 1

h₀ = 0     (hidden state similar to short-term memory IN RNN)
u = [2, 3, 1]

We have the forumlas
hₜ₊₁ = Ahₜ + Bxₜ
yₜ = Chₜ + Dxₜ

And we plug in our values:
       A   h₀     B   x₁
h₁ = (0.5 * 0) + (1 * 2) = 2
h₁ = 2 ---+
          ↓
      C   h₁    D   x₁
y₁ = (2 * 2) + (1 * 2)   = 4
y₁ = 4

That was one forward pass. The hidden state (h₁) has been updated to 2 and
the output (y₁) is 4.
And now we do it again with the new values, notice that h₁ is now 2 instead of
0:
      A    h₁    B    x₂
h₂ = (0.5 * 2) + (1 * 3) = 4
h₂ = 4 ---+
          ↓
      C  h₂     D  x₃
y₂ = (2 * 4) + (1 * 3)   = 11
y₂ = 11

      A    h₂     B  x₃
x₃ = (0.5 * 4) + (1 * 1) = 3
x₃ = 3 ---+
          ↓
      C  h₃     D  x₃ 
y₃ = (2 * 3) + (1 * 1)   = 7
y₃ = 7
```
Each input token updates the state of the system based on the previous state
and the current input.

h is similar to the hidden state in RNNs and allows information to be passed
from one step to the next. y is the output of the system at each time step.

__wip__

### SSM as convolutional
I've read that SSM can be seen/used as a convolutional layer like in a
[CNN](./cnn.md) but I'm not sure how this works.

If we look diagram from above it looks like this:
```
    +---+      +---+       +---+
----| U |------| X |---+---| y |----------------------------->
    +---+  ↑   +---+   |   +---+
           |   +---+   |
           +---| A |---+
               +---+
```
And recall that a convolution network will use a matrix called a convolution or
filter to perform the dot product with the input sequence: 
```
   +--------------+
   |        |  |  |
   |        ------|    C = Convolution/Filter
   |   C    |  |  |        +--+--+--+    +--+--+--+
   |        ------|    C = |  |  |  |    |  |  |  |
   |        |  |  |        |--+--+--|    |--+--+--|    +--+
   |--------------|        |  |  |  | .  |  |  |  | =  |  |
   |  |  |  |  |  |        |--+--+--|    |--+--+--|    +--+
   |--------------|        |  |  |  |    |  |  |  |
   |  |  |  |  |  |        +--------+    +--------+
   |--------------|
   |  |  |  |  |  |
   +--------------+
```
In this case I think the A weight matrix can be thought of as analogous to a
convolution filter in CNN. But this filter in SSN is applied accross time and to
spacially, that is moving accross the height and width. The state X at any point
in time is influenced by the previous state and the current input. This is
similar to how a convolutional filter influences the output based on the input
region that it covers.
