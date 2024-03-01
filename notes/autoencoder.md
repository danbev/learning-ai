## AutoEncoder
Are a type of generation model which uses a neural network which take the input
and reduce its dimensionality but still retain the important information.

They constist of three components:
* Endcoder
* Code
* Decoder

```
  +--+
  |  |     +---\                /---+    +--+
  |  |     |    \              /    |    |  |
  |  |     |     |    +--+    |     |    |  |
  |  |---->|     |--->|  |--->|     |--->|  |
  |  |     |     |    |  |    |     |    |  |
  |  |     |    /     +--+     \    |    |  |
  |  |     +---/                \---+    +--+
  |  |
  +--+

 Input    Encoder    Code      Decoder    Output
                   (bottleneck)
                   (latent rep)
```
Now, the input could be an image and the output the same image which might sound
strange but the idea is that the decoder will be trained to reconstruct the
original input from the code. The code is a lower-dimensional representation of
the input. So that would be used for compression for example. where we use the
output of the encoder, the code vector, and send it over the network to receiver
how uses a decoder to reconstruct the original input.
But we could also imagine where we have a blurry input image and the output
image (the target image that we compare with and calculate the loss of) is 
the clear image. The network would then be training to reduce the blurriness of
the input image.

So lets make this more concrete. If we have an image as input this will be
converted into a high-dimensional vector (the pixels of the image). This vector
is then passed through to the encoder. The encoder will the pass this through
its layers and reduc/compress the data into a lower-dimensional representation.
This process involves nonlinear transformations that aim to keep the most
significant features of the input.

The code might the look like this for the first image (lets say it is of a cat):
```
 [1.2]
 [3.4]
 [5.6]
 [7.8]
```
Now, we do the same for an other cat image and it produces this code:
```
 [8.7]
 [6.5]
 [4.3]
 [2.1]
```
Notice that the codes are very different, but they are both cat images. This is
because the autoencoder is only concerned with keeping the most relevant
information from the image and not concerned with the distribution of values in
the latent space. There is no semantic meaning/relationship to the values in the
latent space and the values are just the optimal values to represent the
important inforation in the input image.

So lets try to sample from the latent space above and see what the real issue
is with doing this.
We can try to interpolate between them and see what we get:
```
New vector = ([1.2, 3.4, 5.6, 7.8] + [8.7, 6.5, 4.3, 2.1]) / 2
           = [4.95, 4.95, 4.95, 4.95]
```
We would then pass this new vector through the decoder to get the new image. We
might expect to get a blend of the two images but recall that these values are
just the optimal values to represent the important inforation in the input image
(the higher dimensional vector). So there is no guarantee that taking the mid-
point between these two vectors will give us a valid cat image (or any image at
all). We might just/probably only get noise.


The encoder part of the autoencoder takes an input (e.g., an image) and
compresses it into a lower-dimensional latent representation. This process
involves reducing the dimensions and capturing the most relevant features of
the data in this compressed form.

The decoder then takes this latent representation and attempts to reconstruct
the original input from it. This involves increasing the dimensions from the
latent space back to the original input space.

The reconstruction is then compared to the original input, and a loss is
calculated (e.g., mean squared error for continuous data or binary cross-entropy
for binary data). The goal is to minimize this loss, making the reconstructed
output as close as possible to the original input. This is a reconstruction and
might be lossy reconstruction of the original.

This can be used for compression where we might compress an image into the 
latent space vector which we then send to some other place, like over a
network, and the other side has a decoder which can then reconstruct the image
back from the latent space vector.

General autoencoders cannot generate new data samples, they are mainly used for
the generation of the latent space. This latent space is not structured in a 
way that ensures that all regions within it correspond to valid output. If we
generate from this space, we will most often get noise. To be able to generate
new sample of a specific type of data, like cat images, we also need have some
structure of the latent space to allow for sampling from it and not getting
noise which is what Variational Autoencoders (VAE) help with.

This can be expressed as we want to find the x vector (think of this as an
image) using the code/z vector:
```
p(x) = ∫ p(x|z) p(z) dz
```
Now this is theorectically possible but in practice it is intractable (not
reaonable to do, like guessing a password) because of the size of the latent
space z. In autoencoders p(x|z) represents the models ability to reconstruct x
from the latent representation z.

The above can also be written using the probabilty:
```
        p(x,z)
 p(x) = -------
        p(z|x)
```
Again we are trying to find the x vector (think of this as an image) just like
above. Notice that the denominator contains p(z|x) which represents the models
ability to generate the latent representation z from the input x. This is not
a value that we have which is simliar to the above situation where it is
intractable to compute.

Why is this intractable?  

Lets take a concrete example with z having 6 elements and lets calculate 
p(x|z) for each value of z:
```
z = {1, 2, 3, 4, 5, 6}

p(x | z=1) = 0.1
p(x | z=2) = 0.2
p(x | z=3) = 0.15
p(x | z=4) = 0.25
p(x | z=5) = 0.05
p(x | z=6) = 0.25
```
The marginal probability can be found by:
```
       6
p(x) = ∑   p(x|z) * p(z)
       z=1
```
And if each value of z is equally likely that beens p(z) = 1/6, so we would sum
all the values (moving the multiplication of 1/6 out of the summation "loop"):
```
p(x) = (0.1 + 0.2 + 0.15 + 0.25 + 0.05 + 0.25) * 1/6
     = 0.166666
```
So what is this number telling us?  
Considering all possible underlying states of z and their probabilities, the
overall chance of observing x is 0.166666, or about 16,7%.
Now, in practice the latent space z is much larger than 6 and this becomes
intractable to compute.


### Undercomplete autoencoder
Can be used to generate a compressed representation of the input data.

### Sparse autoencoder
This type of network is regularized to have a small number of active neurons in
the hidden layer. A penalty is added to some of the neurons making them less or
completely in-active. So this is where sparse comes from.

This can be used to learn a more compact representation of the input data.

### Variational autoencoder (VAE)
We previously mentioned using a traditional autoencoder to generate new samples
from the latent space would result in noise and we discussed the reason for this.

What a VAE does is it add structure to the latent space to allow for sampling
from it.

The architecture of a VAE looks something like this:
```
  +--+
  |  |     +---\                        /---+    +--+
  |  |     |    \                      /    |    |  |
  |  |     |     |  +--+      +--+    |     |    |  |
  |  |---->|     |->|μ |--+   |  |--->|     |--->|  |
  |  |     |     |  +--+  |-->|z |    |     |    |  |
  |  |     |    /   +--+  |   +--+    \     |    |  |
  |  |     +---/  ->|σ |--+            \---+     +--+
  |  |              +--+
  +--+

 Input    Encoder  reparam.  sampled   Decoder    Output
                             latent
                             vector
```

Lets take our example from above (at least the first input):
```
 [1.2]
 [3.4]
 [5.6]
 [7.8]
```
Now, with a VAE the output of the encoder might look like this:
```
mean (μ)     = [1.0] (mean for dimension 1)
               [3.0] (mean for dimension 2)
               [5.0] (mean for dimension 3)
               [7.0] (mean for dimension 4)

veriance (σ) = [0.5] (variance for dimension 1)
               [0.5] (variance for dimension 2)
               [0.5] (variance for dimension 3)
               [0.5] (variance for dimension 4)
```
So notice that the encoder will output two vectors, the mean and the variance
in contrast to autoencoders which output the code/z vector directly.

So VAE will generate the `code/z` vector using the mean and standard deviation
by randomly sample from a distribution specified by the means and variances.
So at this point we only have two vectors of the same length, the mean and the
variance (just to be clear that the input x is no longer used after this point
in the encoder). We will then sample from a normal/gaussian distribution using
the mean and variance to generate the `code/z` vector:
```
 z = [μ₁ + σ₁ * ε₁]    where ε₁ ~ N(0, 1)
     [μ₂ + σ₂ * ε₂]    where ε₂ ~ N(0, 1)
     [μ₃ + σ₃ * ε₃]    where ε₃ ~ N(0, 1)
     [μ₄ + σ₄ * ε₄]    where ε₄ ~ N(0, 1)
```
To clarify what is happing here is that for each dimesion, which is 4 in our
case, we we going to sample a value from the normal distribution which has a
mean of 0 and a standard deviation of 1. So at this point we have a random value
from the normal guassian distribution nothing else.

We then scale this value by the standard deviation for that dimension which was
learned during training specifically for this dimension/feature. And this is
scaling the standard deviation (how much the data is scattered around the mean),
so this is scaling it from the standard deviation of the normal distribution
which is 1. Then the learned mean value is added, which shifts the value to be
centered around the mean of the learned distribution.

This process is called reparameterization.
```
Sampled Code Vector = [0.9, 2.8, 5.1, 7.2]
```
Now, lets take a look at a second image also of a cat but slightly changed,
perhaps the lighting is changed or something):
```
 mage 2 = [1.5, 3.5, 5.5, 7.5]

 code   = [1.4, 3.6, 5.7, 7.3]
```
Now, we can try interpolating these two codes and see what we get:
```
  code image1 = [0.9, 2.8, 5.1, 7.2]
  code image2 = [1.4, 3.6, 5.7, 7.3]

  new vector  = ([0.9, 2.8, 5.1, 7.2] + [1.4, 3.6, 5.7, 7.3]) / 2
              = [1.15, 3.2, 5.4, 7.25]
```
Passing this interpolated vector through the decoder will generate a new image.
Since both original images were of cats and the latent space of the VAE is
structured to ensure that similar inputs map to overlapping distributions, this
new image should realistically represent a cat. The generated image might blend
features from both original cat images, such as pose or lighting, reflecting the
intermediate nature of the interpolated vector.

And we could also generate new cat images by sampling from the distribution:
(recall that sampling involved randomness so the generated images will not be
the same each time we sample from the distribution)
```
mean (μ) = [1.0, 3.0, 5.0, 7.0]
Variance (σ²) = [0.5, 0.5, 0.5, 0.5] ([0.707, 0.707, 0.707, 0.707])

 N(1.0, 0.707) = 1.2
 N(3.0, 0.707) = 2.9
 N(5.0, 0.707) = 4.8
 N(7.0, 0.707) = 7.1

Latent vector (code):
 [1.2]
 [2.9]
 [4.8]
 [7.1]
```
This would then be passed through the decoder to generate a new cat image.

The reparameterization trick is used to ensure that the model is differentiable
and can be trained using backpropagation:
```
z = μ + σ * ε

ε ~ N(0, 1)
```
Where ε is a random sample from the standard normal distribution.  The mean and
variance are learned during training, but ε is fixed, we are randomly sampling
from it but is is not backpropagated through.

So, for the encoding part what the VAE is doing, what it is trained to do that
is, is to predict z given an input image x:
```
p(z|x)
```
And the decoder does the opposite, it predicts x given z, so it is given also
latent vector z and it will use that to predict the input image x:
```
p(x|z)
```

```
 p(x) = ∫ p(x|z) dz
```
This can also be expressed using the chain rule:
```
        p(x,z)
 p(x) = -------
        p(z|x)
```
Where p(x) is the marginal likelihood of the data and recall that p(x) means the
probability of observing x given z. Marginal means to calculate
the probability of x without considering a specific value of z so we are
calculating all possible values of z, effectivly averaging or summing over the
entire range of z. This is intractable to compute in practice.

```
z = {1, 2, 3, 4, 5, 6}

p(x|z=1) = 0.1
p(x|z=2) = 0.2
p(x|z=3) = 0.15
p(x|z=4) = 0.25
p(x|z=5) = 0.05
p(x|z=6) = 0.25
```
The marginal probability can be found by:
```
       6
p(x) = ∑   p(x|z) * p(z)
       z=1
```
And if each value of z is equally likely that beens p(z) = 1/6, so we would sum
all the values (moving the multiplication of 1/6 out of the summation "loop"):
```
p(x) = (0.1 + 0.2 + 0.15 + 0.25 + 0.05 + 0.25) * 1/6
     = 0.166666
```
So what is this number telling us?  
Considering all possible underlying states of z and their probabilities, the
overall chance of observing x is 0.166666, or about 16,7%.

