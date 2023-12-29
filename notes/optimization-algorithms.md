## Opimization Algorithms
Describes different optimization algorithms used in deep learning.

### Frequent vs Infrequent Parameters
To understand this I think an example is most useful. Take a text
classification task in NLP where we have a vocabulary of 10,000 words. Each
word in the vocabulary is a parameter/weight in the model.

In any given text sample used for training only a small fraction of these words
will actually be used. In most languages there are a small number of words that
are used very frequently and a large number of words that are used very
infrequently. Words like "the", "and", "a", "is", etc. are used very frequently
and the weights that correspond to these words will be updated more frequently
(they are always part of the gradient calculation).

With SGD all weights are updated uniformly. So the weights that correspond to
frequent words might get overemphasized and the weights that correspond to
infrequent words, which might be rare but also might actually have more
importance (carry more information because they are rare), will not be updated
with enough importance (perhaps they should be made more important as less
frequent words often convey more information).

In adaptive learning rate methods (like AdaGrad, RMSprop, or Adam), the learning
rates are adjusted based on the update history of each weight. Weights for rare
words get larger updates when they do appear, helping the model to pay attention
to these infrequent but potentially significant features.

### Batch Gradient Decent
Lets take the following example:
```
area         bedrooms     price
2600         3            550000 
3000         4            565000
3200         3            610000
3600         3            595000
4000         5            760000
4100         6            810000

   price = w₁ * area + w₂ * bedrooms + b
```
So we would have take the first entry and perform the following calculation:
```
w₁ = 1, w₂ = 1, b = 1

   price = w₁ * 2600 + w₂ * 3 + b
   price = 2600 + 3 + 1
   price = 2604
```
We would then calculate the error for this prediction:
```
error = (actual - predicted)²
error₁ = (550000 - 2604)²
```
And we would to this for all our samples, which are 6 in this case:
```
total_error = error₁ + error₂ + error₃ + error₄ + error₅ + error₆

MSE = 1/n * total_error
MSE = 1/6 * total_error
```
What we then do is that we take the partial derivative of the MSE with respect
to each of the weights and the bias. This gives us the gradient for each of
the weights and the bias. We then update the weights and the bias with the
following formula:
```
w₁ = w₁ - α * ∂MSE/∂w₁
w₂ = w₂ - α * ∂MSE/∂w₂
b = b - α * ∂MSE/∂b
```
Where α is the learning rate. This is the rate at which we update the weights
and the bias. If we set α to a high value we might overshoot the minimum and
if we set it to a low value it will take a long time to converge.
We takes these new values for the weights and the bias and repeat the process
until we reach a minimum. Each one of these iterations is called an epoch.
We keep doing this until we reach an acceptable error rate/loss, which should be
as close to zero as possible. When this is achieved we values in the weights
and bias which should be correct enough that we can predict the price of a
house given the area and the number of bedrooms. The above is called batch 
gradient decent.
Notice that we went through all the training samples and calculated the error,
which means calculating the gradient for all the samples. In our case this is
not really a problem which such as small data but if we have millions of
samples this can be very inefficient. Also in this example we only have two
features (area and bedrooms) but in real life we might have hundreds of
features which means that we would have to calculate the partial derivative
for each of the features which can be very time consuming.

Gradient decent formula (same as above but with different notation as it might
be written like this in places):
```
θ_new = θ_old - α * ∇J(θ)

θ = represents the parameters of funtion we are trying to optimize which are
    the weights.
α = learning rate.
- = the opposite of the gradient, which is because we want to go down the hill.
∇ = Nabla symbol which is the gradient.
∇(J)(θ) = the gradient of the cost function J with respect to the parameters θ.
          This tells us the slope of the hill under our feet.
```
The process is an iterative process where we start with an initial guess for
the θ, and then repetedly update it until theta converges to a value that
minimizes the cost function J(θ). Convergence means that the values of the
parameters θ stop changing significantly with each iteration, or they change
within a very small predefined threshold.
When we can't go downward anymore we are done or after a specific number of
iterations (or perhaps a combination of both).

### Stochastic Gradient Descent (SGD)
Building off of the previous section SGD tried to address the issue with having
to calculate the error for all the samples in the training dataset. In this case
we only take one sample at a time and calculate the error and update the weights
and bias. This is called an iteration. We then repeat this process until we
reach a minimum. This is called stochastic gradient decent. The problem with
this approach is that it is very noisy and it might not converge to a minimum
but instead bounce around the minimum. This is because we are only taking one
sample at a time and the error for that sample might be very high or very low
which means that the gradient will be very high or very low. This will cause
the weights and bias to jump around the minimum. This is not a problem if we
have a lot of samples but if we have a small dataset this can be a problem.

### Stochastic Gradient Descent with Mini-Batches
This builds upon SGD but adds small batches of values to calculate instead of
a single value. The ideas is to avoid the noise/bouncing of SGD and still be
not have to calculate the error for all the samples. So instead of taking one
sample at a time we take a small batch.


### Stochastic Gradient Descent with Momentum
The idea of momentum in the context of SGD comes from physics, particularly the
concept of momentum in motion. It's like a ball rolling downhill; the momentum
term increases the speed of the descent. So instead of just taking the current
gradient into account it also take the previous gradient into account. So it
needs to keep a vector of values that contain a combination of the gradients
from current step and the velocity of the previous step, scaled by a parameter
known as the momentum coefficient.
```
vₜ= the velocity at time step t.
μ = the momentum coefficient (between 0 - 1)
α = the learning rate
```
Note that if the momemtum is 0 then this is just like standard SGD. And if it is
1 it is like a ball rocking back and forth. A value of 0.8-0.9 is usually a good
value for the momentum coefficient which is like having a little friction so
that the "ball" eventually slows down and stops.

### AdaGrad
Adaptive Gradient Algorithm (AdaGrad) is an algorithm for gradient-based
optimization that adapts the learning rate to the parameters, performing larger
updates for infrequent parameters, and smaller updates for frequent parameters.

The more you have updated a feature already the less you will update it in the
future.

One issue with AdaGrad is that of diminishing learning rates, which prevents the
method from converging to the global optimum.

### RMSProp
Root mean squared (SMS) propagation is an adaptive rate optimization simliar to
AdaGrad and intented to address the issue of diminishing learning rates.
Also a problem with AdaGrad is that it is slow the sum of gradients squared only
grows and never shrinks.

### Adam
Adaptive Movement Estimation (Adam) is an adaptive learning rate optimization
algorithm that's been designed specifically for training deep neural networks.
It takes ideas from both RMSProp and Momentum.


The words that are not used frequently will have a gradient of
zero.

### First-order Optimization Algorithms
These are algorithms that use the first derivative (gradient) of the objective
function, and examples of these are the ones we discussed above, like SGD,
AdaGrad, RMSProp, and Adam. These are effient to compute but because the only
use the slope/gradient they can sometimes lead to less accurate steps in complex
landscapes. 

### Second-order Optimization Algorithms
These are algorithms that use the second derivative, a Hessian matrix that
provides information about the curvature of the function.
These are more accurate but are more expensive to compute because calculating
the Hessian matrix is more complex. But since this method takes the curvature
into consideration, and not just the slope, it can take more effient steps.
In first-order optimization algorithms we specify a learning rate, but in
second-order optimization algorithms we don't need to specify a learning rate
because the Hessian matrix provides information about the curvature of the
function and the step size is automatically adjusted based on this information.
This can lead to faster convergence and removes the burden of learning rate
tuning. The algorithm dynamically adjusts the step size based on the local
curvature of the loss function.

Examples of second-order optimization algorithms are Newton's method,
BFSG (Broyden-Fletcher-Goldfarb-Shanno), and L-BFGS (Limited-memory BFGS).
