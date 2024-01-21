## Mamba: Linear-Time Sequence Modeling with Selective State Space
Mamba is a type of selective state space model (SSSM), `sss` like a snake and
therefor named Mamba I've read. I've got some notes on
[SSM](./state-space-models.md).

Mamba is said that it might be just as influential as the transformer
architecture but this is left to be seen.

One of the authors is Tri Dao, was also involved in the developement of 
[Flash Attention](https://github.com/Dao-AILab/flash-attention) and there one
part of Mamba is taking advantage of the GPU hardware.

So we have transformers which I thought were the absolute latest and advanced
as they seem to be used all over the place. Transformers are effecient at
training as they can be parallelized, incontrast to RNNs which are sequential.

Now, the issue with transformers is that they don't scale to long sequeces which
is because the the self attention mechanism is quadratic in the sequence length.
Every token has to attend to every other token in a sequenc (n²). So if we
have 40 tokens that means 1600 attention operations, which means more
computation and this just increases the longer the input sequence it. In this
respect RNNs are more performant as they don't have the quadratic scaling issue
that the self attention mechanism has (but do have other issues).

The core of Mamba is state space models (SSMs). Before we go further it might
make sense to review [RNNs](./rnn.md) and [SSMs](./state-space-models.md).

Selective state space models, which Mamaba is a type of, give us a linear
recurrent network simliar to RRNs, but also have the fast training that we gets
from transformers. So we get the best of both worlds.

One major difference with state space models is that they have state which is
not something the transformers have. So transformers don't have an intrinsic
state which gets updated as the model processes a sequence. But neural networks
like RNNs do have state but recall that they process the input sequence

### Selective State Space Models
Selective State Space is a type of state space and a state space is defined
by two funcions:
```
h'(t) = Ah(t) + Bx(t)              (state equation)
yₜ = Ch(t) + Dx(t)  (output equation) (Dx is not referred to in the paper)

h ∈ Rⁿ  is the hidden state
x ∈ R¹  is the input
y ∈ R¹  is the output
A ∈ Rⁿ×ⁿ is the state transition matrix
B ∈ R¹×ⁿ is the input matrix
C ∈ Rⁿ×¹ is the output matrix
```
Now, above we have the delta, A, and b, which are continuous values as per
the definition of a state space model. This makes sense if we think about it as
this is not specific to neural networks or even computers.
Think about an analog system, for example an IoT device that reads the
temperature from a sensor connected to it. To process this signal it needs
to be converted into digital form. A simliar thing needs to be done in this case
as we can't use continous signals with computers, just like an IoT can't process
an analog signal directly. So we need to convert into descrete time steps,
similar to how an Analog-to-Digital Converter ([ADC]) would convert the signal
into quantized values. This step is called discretization in the state space
model.

[ADC]: https://github.com/danbev/learning-iot/tree/master?tab=readme-ov-file#analog-to-digital-converter-adc

### Discretization
So we will first discretize the parameters A, and B of the state space model,
which means that we will convert them from continuous values to discrete values.

I think there are multiple methods/ways to do this but the paper mentions
the bilinear (Tustin's) transform method which is a method for converting a
continous time system into a discrete time system. It provides a way of
approximating the behaviour of a continuous time system with a discrete time
system.

The S-domain is the continuous time domain and the Z-domain is the discrete
time domain.
The S-plane is used in continuous systems and can be visualized as a x-y plane
but for complex numbers. The x-axis is the real part of the complex number,
and the y-axis is the imaginary part of the complex number: 
```
Imaginary
part         ↑
             |
             |
             |
             |
             +------------------→ Real part


S = σ + jω

σ = real part which represents the decay or growth rate
ω = imaginary part which represents the oscillation rate, 0 = no oscillation
j = √-1 (imaginary unit)
```

And then we have the Z-plane which is used in discrete systems and can be
visualized as a x-y plane but for complex numbers. The x-axis is the real part
of the complex number, and the y-axis is the imaginary part of the complex just
like in the S-plane:
```
Imaginary
part         ↑
             |
             |
             |
             |
             +------------------→ Real part

z = re^(jθ)
re = magnitude
e = Euler's number
θ = phase angle
j = √-1 (imaginary unit)
```
The Z-plane is often represented in polar coordinates because discrete-time
signals are inherently periodic due to sampling.

So the bilinear transform is taking a point (or points I guess, but lets use one
point for this example) in the S-plane and representing it as a point in the
Z-plane.

Lets take the following point in the S-plane, S = -1 +2j with a sampling
period of T = 1:

![image](./bilinear.png)


```
    2   (z - 1)
S = - * -------
    T   (z + 1)

    1 + (T/2)S
z = ----------
    1 - (T/2)S


    1 + (1/2)(-1 + 2j)
z = ------------------
    1 - (1/2)(-1 + 2j)


z_point = re=-0.07692307692307691, im=0.6153846153846154j
```



_wip_

The bilinear transformation is a method to map the continuous-time domain
(S-plane) to the discrete-time domain (Z-plane)

Billinear transform:
```
    2   (z - 1)
S = - * -------
    T   (z + 1)

Where:
S = the Laplace transform variable
T = the sampling period
z = the z-transform variable
```

When we apply the bilinear transform to the state space model we are
recalculating how the system's state should be updated in descrete time
intervales instead of continuous time intervals.

So instead of the using functions as shown above we concrete values:
```
     _       _ 
hₜ = Ahₜ₋₁ + Bxₜ
yₜ = Chₜ+ Dxₜ
```
Where A_hat and B_hat are:
```
A_hat = (I - Δ/2 A)⁻¹
B_hat = (I - Δ/2 A)⁻¹ ΔB

Δ = the time step, for example if we sample every minute then Δ = 1
I = the identity matrix
A = the state transition matrix
B = the input matrix
```

We can visualize this as
```
    +---+      +---+       +---+
----| x |------| h |---+---| y |----------------------------->
    +---+  ↑   +---+   |   +---+
           |   +---+   |
           +---| A |---+
               +---+
```


### Discretization
Involves converting a continuous function, equation, or model into a discrete
form. In a continuous domain, variables can take on an infinite number of values
within a range. After discretization, these variables can only take on distinct,
separate values.
