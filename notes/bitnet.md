## BitNet 1.58
This is an extension of the original [Bitnet] which just used 1 and -1 as values
(so binary), and extends this to use [1, 0, -1] (so ternery).

Bitnet introduces a concept called 'BitLinear' which is a method for compressing
neural network weights and activations to reduce computational requirements
while attempting to preserve the model's accuracy. 

Normally weights are stored as 32-bit or 16-bit floating point numbers (we also
have quantized versions). BitLinear stores weights as 1-bit values, and
activations as 8-bit values.

The original BitNet converted the weights to 1-bit values [1 or -1] using the
`signum` function which assignes +1 to positive values, and -1 to negative
values.
So first the weights are binarized using the `signum` function which looks like
this:
```
W_f = signum(W - σ)

Where `W_f` is binarized weights matrix, `W` is the original weight, and `σ` is
the mean/centralization factor:
```
     1
α = --- Σ W_i_j
     nm

nm = the dimension of the weight matrix, and `W_i_j` is the weight at position
`i, j` in the matrix. So this is just the mean.

With only values of 1 and -1 the matrix operations like multiplication becomes
addition. Think about taking a number and multiplying it by 1, that just gives
the same number. And multiplying by -1 just gives the negative of the number.
So not only does storing the weight in smaller sizes in memory reduce the memory
footprint but this also inproved the computational cost where we today need org
at least improve performance using GPUs.

So that was for the weights, but what about the activations?   
The activations or outputs of the neural network layers are quantized to 8-bit
precision (or actually b bit precision):
```
                          Q_b
xₑ = Quant(x) = Clip( x * ---, Q_b + ε, -Q_b - ε)
                           γ
xₑ  = the quantized value.
x   = the original activation values (a matrix)
Q_b = 2^(b-1)
γ   = gamma is absolute maximum value of the original activation values x.
ε   = is to prevent overflow.
```
So lets say we want to quantize the activations to 8 bit this would mean that
Qb = 2^7 = 128. So the range of the quantized values would be -128 to 127.
```
                          128
xₑ = Quant(x) = Clip( x * ---, 128 + ε, -128 - ε)
                           γ
```
Notice that the first argument to the clip function are scaled values by 128/γ.
And recall that gamma is the absolute maximum value of the original activation
values x. 
And the `Clip` function is defined as:
```
Clip(x, a, b) = max(a, min(b, x))
```
Just to make this a little more concrete, lets make up some values:
```
γ = 512
ε = 000.1

xᵢ = 224
x_e_1 = 224/512 = 0.4375
Clip( 0.4375, 128 + 0.1, -128 - 0.1) = max(0.4375, min(128.1, -128.1)) = 0.4375
```

Note that this is done during training and is not a post training quantization.

[bitnet]: https://arxiv.org/pdf/2310.11453.pdf
[bitnet 1.58]: https://arxiv.org/pdf/2402.17764.pdf
