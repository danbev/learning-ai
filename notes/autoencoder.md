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

So lets make this more concrete. If we have an image as input this will be
converted into a high-dimensional vector (the pixels of the image). This vector
is then passed through to the encoder. The encoder will the pass this through it
layers and reducing/compressing the data into a lower-dimensional
representation. This process involves nonlinear transformations that aim to
keep the most significant features of the input.

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
the latent space.
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
generate from this space, we will most often get noise.  To be able to generate
new sample of a specific type of data, like cat images, we also need have some
structure of the latent space to allow for sampling from
it and not getting noise which is what Variational Autoencoders (VAE) help with.

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
What VAE does is it add structure to the latent space to allow for sampling from
it.

Lets take our example from above (at least the first input):
```
 [1.2]
 [3.4]
 [5.6]
 [7.8]
```
Now, with VAE it might generate a code (vector) that looks something like this:


```
mean (μ) = [1.0, 3.0, 5.0, 7.0]
variability (σ) = [0.5, 0.5, 0.5, 0.5]
```
So VAE will generate the `code` using this distribution the VAE randomly samples
from the distribution by the above means and variances. This process is called
reparameterization
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

