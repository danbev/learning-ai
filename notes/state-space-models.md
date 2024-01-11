## State Space Models (SSM)
State of a system is the set of variables which as some point in time, together
with the inputs to the system, completely determines the behaviour of the
system. SSM is a broad class of models that can be used to describe the
behaviour of a system over time and used in various domains such as control
theory, robotics, and machine learning.

#### Structured State Space Models (S4)
Is a type of SSM that is speciallized for the domain of deep learning and
sequential modeling.

Simliar to RRNs, S4 models are designed to process sequential data specifically
where the order of input matters (nlp, audio signal processing).

Both architectures maintain a form of state across time steps. In RNNs, this is
the hidden state, while in S4 models, it's the state vector `x`.

In RNNs, the parameters of the model are the weights and biases of the network,
whereas in S4 models, the parameters are the state transition matrix `A`, the
input matrix `B`, the output matrix `C`, and the feedthrough matrix `D`.

The forula for S4 models is:
```
xₜ₊₁ = Axₜ + Buₜ
yₜ = Cxₜ + Duₜ

A = state transition matrix
B = input matrix
C = output matrix
D = feedthrough matrix
```

Lets take an example, the inputs we have are tokens but well represent them
as [u₁, u₂, u₃] and they could for example represent words in a sentence:
```
A = 0.5
B = 1
C = 2
D = 1

x₀ = 0     (hidden state like short-term memory?) 
u = [2, 3, 1]

       A   x₀     B   u₁
x₁ = (0.5 * 0) + (1 * 2) = 2
x₁ = 2
       
      C   x₁    D   u₁
y₁ = (2 * 2) + (1 * 2)   = 6
y₁ = 6

      A    x₁    B    u₂
x₂ = (0.5 * 2) + (1 * 3) = 4
x₂ = 4

      C  x₂     D  u₃
y₂ = (2 * 4) + (1 * 3)   = 11
y₂ = 11

      A    x₂     B  u₃
x₃ = (0.5 * 4) + (1 * 1) = 3
x₃ = 3

      C  x₃     D  u₃ 
y₃ = (2 * 3) + (1 * 1)   = 7
y₃ = 7
```
Each input token updates the state of the system based on the previous state
and the current input.

x is similar to the hidden state in RNNs and allows information to be passed
from one step to the next.
__wip__
