## Residual Connections
A residual connections, or skip connections, is a way to make sure that a layer
in a neural network does not "forget" the original input. For example in a
transformer model where the input to the later is added output of the layer. The
way I think about a layer is that it moves the token embeddings around in the
embedding space. So a layer will move the token embeddings slightly perhaps using
various operations. We then perform a matrix addition (element wise), so each
entry in the matrix has the original value added to itself. So this "feels" like
it is kind making sure that the token embeddings does not get moved too much
relative to the original value.

It's like allowing each layer to transform the input it gets, and instead of
replacing the input (which will become the input to the next layer), it add those
changes to the input it received. So this avoids overwriting/replacing the input
and instead lets the layer make adjustments to it.

That is one aspect of this, preserving information, the other is technical issue
related to training which is the vanishing gradient problem. The gradients are
updated using backpropagation starting from the outer most layer and then it update
the weights of each layer going backwards.

Lets take a concrete example:
```console
Network one input, two hidden layers, and one output:

x_0 : input
x_1 : output of layer 1
x_2 : output of layer 2
x_3 : final output
```

We want to calculate the gradient of the output (x_3) with respect to the input
(x_0):
```console
∂x_3
----
∂x_0
```

In a standard network, each node is a function of the previous one.
```console
x_1 = F(x_0)
x_2 = G(x_1)
x_3 = H(x_2)
```
To find how much x_0 affects x_3, the Chain Rule tells us to multiply the local
derivatives of every step along the way:
```console
∂x_3
---- = H'(x_2) * G'(x_1) * F'(x_0)
∂x_0   layer 3   layer 2   layer 1

How much does the output (x_3) respond to changes in the input (x_0).

If the layers are "lazy" or initialized with small weights:
Gradient = 0.1 * 0.1 * 0.1 = 0.001
```

With a skip connection this becomes:
```console
x_1 = x_0 + F(x_0)
x_2 = x_1 + G(x_1)
x_3 = x_2 + H(x_2)
```

Lets look at the chain rule for the first node (`x_1 = x_0 + F(x_0)`):
```console
∂x_1     ∂(x_0)   ∂(F(x_0))
---- =   -----  + ---------
∂x_0     ∂x_0       ∂x_0
```
The derivative of `x_0` with respect to itself is 1, so we have:
```console
∂x_1       ∂(F(x_0))
---- = 1 + ---------
∂x_0         ∂x_0
```
And if we look at the whole chain:
```console
∂x_3
---- = (1 + H'(x_2)) * (1 + G'(x_1)) * (1 + F'(x_0))
∂x_0   

Total Gradient = (1 + 0.1) * (1 + 0.1) * (1 + 0.1)
Total Gradient = 1.1 * 1.1 * 1.1 = 1.331
```
Instead of shrinking to 0.001, the signal stayed strong at 1.331.
```

(1 + H'(x_2)) * (1 + G'(x_1)) * (1 + F'(x_0))
    ↓

1 * 1 * 1 + other terms
    ↓
    1     + other terms
```
That 1 is the "superhighway". It mathematically guarantees that there is a
"baseline" gradient flow. Even if every single layer (F, G, H) has a gradient of
zero (meaning the layers are dead/doing nothing), the total gradient is still 1.

Just to recap, the forward pass outputs a prediction and during training we
calculate a loss based on how far off that prediction is from the true value. Using
this loss we want to know how much to adjust each weight in the network to reduce
the loss.

Flow:
```console
x_0 --> [network] -> x_3 --> [loss function] -> L (loss)
```

How do we change the output (x_3) to reduce the loss (L)?  
The Chain Rule actually starts with the Loss (L).

```
Total Gradient = (Gradient of Loss vs Output) * (Gradient of Output vs Input)
```

Let's say our Loss Gradient is -2 (The model made a mistake).

1. Without Skip Connection:
```
   Total Gradient = -2 * 0.001 = -0.002
   (The error signal vanishes. The first layer assumes everything is fine.)
```

2. With Skip Connection:
```
   Total Gradient = -2 * 1.331 = -2.662
   (The error signal is preserved and even amplified. The first layer knows
    exactly how to fix the mistake.)
```

We want to calculate the gradient of the output (x_3) with respect to the input
(x_0):
```console
∂L      ∂(L)    ∂x_3
---   = ----- * ----
∂x_0    ∂(x_3)  ∂x_0
```

Lets say that the true/known value is 10 but the model outputs 8 (x_3):
```console
Prediction (x_3) = 8
Target           = 10
Loss Function    = (Target - Prediction)^2
Loss             = (10 - 8)^2 = 4
```

We need to find ∂L/∂x_3. This asks: "How does the Loss change if the Prediction
(x_3) changes?  
```console
Equation: L = (10 - x_3)^2         (Loss Function)
Derivative: 2 * (10 - x_3) * (-1)  (The -1 comes from the negative sign on x_3)
Simplified: 2 * (10 - x_3)
```
Plug in our numbers (x_3 = 8):
```
Gradient = -2 * (10 - 8)
Gradient = -2 * 2
Gradient = -4
```
The gradient is -4 and since it is negative it tells us that we need to increase
the input (x_3) to reduce the loss (move from 8 towards 10). The magnitude (4)
tells we are far off so we need a significant adjustment.
TODO: connect this to the actual gradient/slope to clarify further.

The Full Chain (No Skip Connection):
```console
Total Gradient = (Loss Gradient) * (Layer 3) * (Layer 2) * (Layer 1)
Total Gradient = (-4)            * 0.1       * 0.1       * 0.1

Calculation:
-4 * 0.001 = -0.004
```
The input x_0 receives a tiny signal (-0.004). It barely updates. The "message"
from the loss function was lost in transit.

The Full Chain (With Skip Connection):
Recall that the local gradient for a residual block is (1 + layer) = (1 + 0.1) = 1.1
```console
Total Gradient = (Loss Gradient) * (Block 3) * (Block 2) * (Block 1)
Total Gradient = (-4)            * 1.1       * 1.1       * 1.1

Calculation:
-4 * 1.331 = -5.324
```
The input x_0 receives a strong signal (-5.324). And notice that the signal is
actually stronger than the original error (-4).

So this example is only adjusting the weights for the first node and we would
continue doing this for the others as well where their weight might be adjust
less/more depending on their current values and the gradient that is calculated.
The first node (the input layer) is the layer that is most susceptible to the
vanishing gradient problem because the gradient has to flow through all the
layers to reach it. The later layers are closer to the loss function so they get
a stronger signal. For example x_3 would get the full -4 signal since it is the
output layer.

We update the weighs using the following formula:
```console
New Weight = Old Weight - (Learning Rate * Gradient * Input Value)
```
The Input Value: Was this neuron actually active? (If the input was 0, we don't
update the weight because it didn't contribute to the error).

