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
xₜ₊₁ = Axₜ + Buₜ
yₜ = Cxₜ + Duₜ

Xₜ₊₁ = is the state at time t+1
A    = state transition matrix
Xₜ   = is the state at time t
B    = input matrix
Uₜ   = input vector
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

x₀ = 0     (hidden state similar to short-term memory IN RNN)
u = [2, 3, 1]

We have the forumlas
xₜ₊₁ = Axₜ + Buₜ
yₜ = Cxₜ + Duₜ

And we plug in our values:
       A   x₀     B   u₁
x₁ = (0.5 * 0) + (1 * 2) = 2
x₁ = 2 ---+
          ↓
      C   x₁    D   u₁
y₁ = (2 * 2) + (1 * 2)   = 4
y₁ = 4

That was one forward pass. The hidden state (x₁) has been updated to 2 and
the output (y₁) is 4.
And now we do it again with the new values, notice that x₁ is now 2 instead of
0:
      A    x₁    B    u₂
x₂ = (0.5 * 2) + (1 * 3) = 4
x₂ = 4 ---+
          ↓
      C  x₂     D  u₃
y₂ = (2 * 4) + (1 * 3)   = 11
y₂ = 11

      A    x₂     B  u₃
x₃ = (0.5 * 4) + (1 * 1) = 3
x₃ = 3 ---+
          ↓
      C  x₃     D  u₃ 
y₃ = (2 * 3) + (1 * 1)   = 7
y₃ = 7
```
Each input token updates the state of the system based on the previous state
and the current input.

x is similar to the hidden state in RNNs and allows information to be passed
from one step to the next. y is the output of the system at each time step.

__wip__
