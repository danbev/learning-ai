## Diffusion Models
Training in such a model is done by adding noise to an image, a little at a time
in small increments. This is called the forward diffusion process.

### Adding noise to an image
How exactly is noise added to an image?  
This is something that was not clear to me. This section is an attempt to make
this more concrete.

So let say we have a grayscale image of size 3x3. And the values can be between
0-255 (0 = black, 255 = white):
```
[ 100 150 200 ]
[ 120 180 210 ]
[ 130 160 190 ]
```
So that is the original image x₀ that we want to add noise to. Now, we have the
following forumla:
```
xₜ = √(αₜ)x₀ + √(1 - σₜ)ε
aₜ = 0.5
ε  = guassian noise (random values that are normally distributed)
```

Examples of guassian noise:
```
[26 -47  -10]
[17  70  -16]
[-21 29   -8]
```
So for each pixel in the image we will plug that value into the equation:
```
√(0.5) = 0.707 
√(1 - 0.5) = √(0.5) = 0.707

x₀₀ = 0.707 * 100 + 0.707 * 26 = 70.7 + 18.4 = 89.1
x₀₁ = 0.707 * 150 + 0.707 * -47 = 106.05 - 33.29 = 72.76
x₀₂ = 0.707 * 200 + 0.707 * -10 = 141.4 - 7.07 = 134.33
x₁₀ = 0.707 * 120 + 0.707 * 17 = 84.84 + 11.99 = 96.83
x₁₁ = 0.707 * 180 + 0.707 * 70 = 127.26 + 49.49 = 176.75
x₁₂ = 0.707 * 210 + 0.707 * -16 = 148.47 - 11.31 = 137.16
x₂₀ = 0.707 * 130 + 0.707 * -21 = 91.91 - 14.85 = 77.06
x₂₁ = 0.707 * 160 + 0.707 * 29 = 113.12 + 20.5 = 133.62
x₂₂ = 0.707 * 190 + 0.707 * -8 = 134.23 - 5.65 = 128.58

[ 89  72 134 ]
[ 96 176 137 ]
[ 77 133 128 ]
```
So that was one step, and we can do this for as many steps as we want.

Now, when I saw and read about this below, there is a mention of a forward pass
which is the process of adding noise to an image. I though that only the last
step was recorded and used but all steps are recorded and used as input when
training. And the original image is used as the target (the truth value) when
training. So training it not just done on the final step with the original image
as the target, but on all steps with the original image as the target. That is
how the model learns to remove noise from the image.

The guassian noise is chosen using the guassian distribution or normal
distribution which is a probability distribution that is symmetric about the
mean, showing that data near the mean are more frequent in occurrence than data
far from the mean.  we say we sample from this distrubtion which is basically
drawing random values from the distribution which will follow the probability
distribution function.

Then we have the reverse process which removes noise from an image, which is
called the reverse diffusion process.

Paper: https://arxiv.org/pdf/2209.00796.pdf

### Forward process
The forward process is done by adding noise to an image x₀. We add guassian
noice in iteratively small steps T:
```
αₜ = 1 - βₜ

βₜ = the amount of noise to add at step t.
σₜ = how much of the original image to keep at step t.
```
β is parameter that decides the amount of guassian noise to add at each step.
Its value is decided by the variance schedule which will make the progression
from the original image to the final image smooth, the amount of noice will
increase over time.

We represent the inital image, or input image as x₀.
x₀ is a random variable that follows the distribution function q(x₀) which is the
distribution of the original image. This is written as:
```
x₀ ~ q(x₀)

~     = the random variable on the left is the symbol for "is distributed as"
        follow the probability distribution described on the right side of the
        symbol.
q(x₀) = is the probability distribution function
```
So we have the original image x₀ and we add noise to it in small steps T.
```
x₀ -> x₁ -> x₂ -> ... -> xₜ

-> noise is added.
```
Where xₜ is the final image. Each x variable is a random variable because of
the stochastic nature of the noise.

The can write the probability distribution function as:
```
q(x₁, ..., xₜ|x₀) = ∏ᵢ q(xᵢ|xᵢ-₁)
```
So I read above as `q(x_1, ..., x_t|x_o)` is a function (like in a programming
language) which takes `x₀` (the initial input/state) and the body of the
function will take those values and multiply them together which will then be
then end result which is returned by the function.
We can also write this as:
```
q(xₜ|x₀) = ∏ᵢ q(xᵢ|xᵢ-₁)
```
The following is called the transition kernel, and where we are saying that
the probability of xₜ given x₀ is the product of the probability of the value
of x from the previous step:
```
q(xᵢ|xᵢ-₁)
```
A common choice for the transition kernel is the guassian distribution:
```
q(xᵢ|xᵢ-₁) = N(xₜᵢ; √βₜxₜ₋₁, βₜI)
                   [-------] [--]
                       ↑       ↑
                      mean    variance
Where:
N    = normal distribution
xₜᵢ  = the image at time t (the output)
xₜ₋₁ = the image at time t-1
βₜ   = the amount of noise to add at step t. ε (0, 1)
I    = the identity matrix
```
βₜ ε (0, 1), so the value is dependent on the current time step, and is in or
belongs to the open interval (0, 1) (so values greater than 0 and less than 1).
The notation of N(x ; μ, σ²), here x is the variable of interest, μ is the mean
of the distribution, and σ² is the variance of the distribution.

I'm trying to visualize this and my current mental model is the number line the
for each time step we will have a different mean for the distribution, hence the
distribution changes a little for each time step. And the variance, how much the
samples are spreaad out from the mean is also different for each time step but 
is the same for all dimensions which is the reason for the identity matrix..
And this "should" give different/changing sampled random values as time
progresses. 
Because the mean of the distribution at each time step is dependent on the
previous state, the center (mean) of our normal distribution shifts along the
number line. 

So one pass through the forward process is called a forward pass and the image
will have a little noise added to it for each pass. Now, there is a way to write
this without having show all the steps, and that is to use the following:
```
αₜ = 1 - βₜ
```
And we can define `alpha_bar` which is the cumulative product of all alpha
values for all time steps:
```
-   t
αₜ= Π aₛ
   s=1
```
Recall that the cumulative product of a sequence of numbers is the product of
all numbers up to the current number.
For example for [2, 3, 4]:
```
[2]                     = [2]
[2, 3] = [2 * 3]        = [6]
[2, 3, 4] = [2 * 3 * 4] = [24]
```
So that is what we are doing with the alpha values for all time steps.

So we have the formula from above:
```
q(xᵢ|xᵢ-₁) = N(xₜᵢ; √βₜxₜ₋₁, βₜI)
```
Which we can re-write as:
```
q(xᵢ|xᵢ-₁) = N(xₜᵢ; √αₜxₜ₋₁, αₜI)
           = √1-βₜxₜ₋₁ + √βₜε
```

### Reparameterization trick
When we have an variation autoencoder (VAE) the encoder is not creating the
latent space vector directly, but instead it is sampling from a guassian
distribution with a mean and variance. But when the backpropagation algorithm
is used to train the model, it cannot pass the gradient through a random node.
So the node itself `z` has a value, that is not the problem, the problem is that
the value in this case is produced randomlly by sampling from the normal
distribution N(μ, σ). Let say this was a function n(μ, σ) and it returns
a random value:
```
  float z(float μ, float σ) {
    return rand(μ, σ);
  }
```
If we would try to implement this as a Value/Node in our from-zero-to-hero
project we would not be able to create the backpropagation code because how
could we figure out how a small change to μ or σ would change the output of the
function. We could not, because the output is random.

This randomness prevents the direct backpropagation of gradients through the
sampling process because the derivative of a random sampling operation is not
defined.

This is the problem that the reparameterization trick solves.
So normally we would sample z from a distribution like this:
```
z ~ N(μ, σ)

z is sampled from the distribution N(μ, σ)
μ = mean
σ = standard deviation
```
What we can do is introduce a helper variable ε which is sampled from a standard
normal distribution (mean = 0, standard deviation = 1):
```
ε ~ N(0, 1)
```
This distribution is fixed and does not depend on any of the parameters of the
model. Now we can add this variable as a parameter to z:
```
z = μ + σ * ε
```
So our pseudo code would look like this:
```
  float z(float μ, float σ, float ε) {
    return μ + σ * ε;
  }
```
So the value of epsilon would have been sampled from the standard normal before
calling the z function:
```
float epsilon() {
  return standard_normal_distribution_sample();

}
float z(float μ, float σ, float ε) {
  return μ + σ * ε;
}

// Usage example
float ε = epsilon(); // Sample ε
float sample_z = z(μ, σ, ε); // Compute z
```
So during the forward pass epsilon is sampled, and in the backward pass
(backpropagation) it is a constant value and is scaled by the small
changes/nudges to the parameters mu and sigma.

```
q(xᵢ|xᵢ-₁) = N(xₜᵢ; √βₜxₜ₋₁, βₜI)
```


### Guassian noise
Named after Carl Friedrich Gauss this is a type of noise that is generated by
adding random values that are normally distributed, that is they have a mean of
0 and a standard deviation.

### Standard Deviation
The forumla looks like this:
```
SD = √( ∑|x - μ|² )         
       ---------
          N
x = a value in the data set
N = number of data points in the population
μ = mean of the data set
```
So we are subtracting each data point, x, with the mean, and squaring it
(possibly to avoid negative numbers), and then summing all of them together.
Then we divide that by the total number of data points and then square root
that.

Lets take the following set:
```
{6, 2, 3, 1}
```
First we find the mean:
```
μ = (6 + 2 + 3 + 1) / 4 = 12 / 4 = 3
```
And then:
```
 |6 - 3|² = 3²  = 9
 |2 - 3|² = -1² = 1 
 |3 - 3|² = 0²  = 0
 |1 - 3|² = -2² = 4

 9 + 1 + 0 + 4 = 14

     14/4 = 3.5

√3.5 = 1.870828693
√3.5 ~ 1.870
```
The symbol for standard deviation is σ (sigma).
