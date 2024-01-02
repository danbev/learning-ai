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
So we would take the first entry and perform the following calculation:
```
w₁ = 1, w₂ = 1, b = 1

   price = w₁ * area + w₂ * bedrooms + b
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

MSE = Mean Squared Error
```
What we then do is that we take the partial derivative of the MSE with respect
to each of the weights and the bias. This gives us the gradient for each of
the weights and the bias. We then update the weights and the bias with the
following formula:
```
w₁ = w₁ - α * ∂MSE/∂w₁
w₂ = w₂ - α * ∂MSE/∂w₂
b = b - α * ∂MSE/∂b

∂ = partial derivative
α = learning/step rate
```
Where α is the learning rate. This is the rate at which we update the weights
and the bias. If we set α to a high value we might overshoot the minimum and
if we set it to a low value it will take a long time to converge.

We take these new values for the weights and the bias and repeat the process
until we reach a minimum. Each one of these iterations is called an epoch.

We keep doing this until we reach an acceptable error rate/loss, which should be
as close to zero as possible. When this is achieved the values in the weights
and bias which should be correct enough that we can predict the price of a
house given the area and the number of bedrooms. The above is called batch 
gradient decent.

Notice that we went through all the training samples and calculated the error,
which means calculating the gradient for all the samples. In our case this is
not really a problem which such as small data set, but if we have millions of
samples this can be very inefficient. Also in this example we only have two
features (area and bedrooms) but in real life we might have
hundreds/thousand/millsions of features which means that we would have to
calculate the partial derivative for each of the features which can be very time
consuming.

The following is the Gradient decent formula (same as above but with different
notation as it might be written like this in places):
```
θ_new = θ_old - α * ∇J(θ)

θ = represents the parameters of function we are trying to optimize which are
    the weights.
α = step size/learning rate.
- = the opposite of the gradient, which is because we want to go down the hill.
∇ = Nabla symbol which is the gradient.
∇(J)(θ) = the gradient of the cost function J, with respect to the parameters θ.
          This tells us the slope of the hill under our feet.
```
The process is an iterative process where we start with an initial guess for
θ, and then repetedly update until theta converges to a value that minimizes the
cost function J(θ). Convergence means that the values of the parameters θ stop
changing significantly with each iteration, or they change within a very small
predefined threshold. When we can't go downward anymore we are done or after a
specific number of iterations (or perhaps a combination of both).


### Stochastic Gradient Descent (SGD)
Building off of the previous section SGD tries to address the issue with having
to calculate the error for all the samples in the training dataset. In this case
we only take one random sample at a time and calculate the error and update the
weights and bias. This is called an iteration. We then repeat this process until
we reach a minimum. This is called stochastic (random) gradient decent.

The problem with this approach is that it since we pick a random sample each
interation the calculated gradients can vary significantly between each
iteration. This is because we are only taking one sample at a time and the error
for that sample might be very high or very low which means that the gradient
will be very high or very low. This is called noise. This noise can actually be
beneficial. It adds a degree of randomness that can help the algorithm escape
local minima, potentially leading to better solutions in complex loss
landscapes.

This will cause the weights and bias to jump around the minimum. This is not a
problem if we have a lot of samples but if we have a small dataset this can be
a problem.

### Stochastic Gradient Descent with Mini-Batches
This builds upon SGD but adds small batches of random values to calculate
instead of a single value. The ideas is to avoid the noise/bouncing of SGD and
still be not have to calculate the error for all the samples. So instead of
taking one sample at a time we take a small batch.


### Stochastic Gradient Descent with Momentum
The idea of momentum in the context of SGD comes from physics, particularly the
concept of momentum in motion. It's like a ball rolling down hill; the momentum
term increases the speed of the descent.

So instead of just taking the current gradient into account, it also take the
previous gradient into account. So it needs to keep a vector of values that
contain a combination of the gradients from current step and the velocity of the
previous step, scaled by a parameter known as the momentum coefficient.

First, we update the velocity vector:
```
v = αv − η ∇f(θ) 
v = the velocity vector containing the accumlated gradients updates from the
    previous steps.
α = alpha is the momentum coefficient and this hyperparameter dictates how much
    of the previous velocity vector influences the current update.
η = eta is the learning rate and is also a hyperparameter.
∇f(θ) = ∇(Nabla/Del) is the symbol for the gradient operator. f(θ) is function
        of theta which represents the loss function with respect to the model
        parameters.

θ = theta which are the weights (parameters of the model)
```
And then we update the weights/parameters:
```
θ = θ + v
```
This helpful with saddle points where the slope of the function is zero but it
is not a local max or min. Because of momentum for us to have arrived at the
saddle point we must have been moving down hill and therefor the velocity vector
will case the update to the weights to be larger and we will/can help move past
the saddle point. But it is also possible that the momentum will cause us to
miss a local minimum and the velocity could cause us to overshoot it. This is
controlled by the momentum coefficient alpha (α). There are adaptive learning
rate variants like AdaGrad and Adam that can help with this issue of
overshooting.

### AdaGrad
Adaptive Gradient Algorithm (AdaGrad) is an extension of traditional gradient
decent which adapts the learning rate (the Ada(pt) part of the name) to the
parameters, performing larger updates for infrequent parameters, and smaller
updates for frequent parameters.

AdaGrad modifies the general learning rate at each time step for every
parameter, based on the past gradients that have been computed for that
parameter.

So the learning rate is adapted for each parameter/weight. AdaGrad holds an
array of size N (the number of parameters) and the value in this array, is the
accumulated square of the gradients for each parameter. Recall that we are
talking about stochastic gradient decent, so a random sample will be taken and
the gradient will be calculated for that sample. This gradient will be squared
and then added to the array for that parameter. The next time this parameter is
randomly selected the gradient will be squared and added to the value in the
array for that parameter. 

Each parameters value in G will increase with each iteration since we take the
gradient and square it and then add it to the existing value in G[g]. Something
like following for each step:
```
G[i] += g²
```
The update equation then looks like this:
```
                   η
θ_new = θ_old - ------- * g
                 √(G+ε)  

θ = theta which are the weights (parameters of the model)
η = eta is the global learning rate and is also a hyperparameter.
G = the accumulated square of the gradients for each parameter.
ε = epsilon is a small value added to the denominator to avoid division by zero.
g = the gradient for the current step.
```
Now, like we mentioned above the accumulated square gradient will increase the
more a parameter is updated. But notice that this is then scaled by the global
learning rate (η)/√(G+ε). So the more a parameter is updated the larger its
accumulated gradient value in the G array will be, but this will then be scaled
by the global learning rate. For example:
```
η = 1

  1 / 2.5 = 0.4
  1 / 5.0 = 0.2
```
So the larger the gradient the smaller the value will be smaller which means
that the effective learning rate for that parameter decreases over time.
If a parameter has a large gradient this is like rough terrain and we want to
take smaller steps. If a parameter has a small gradient this is like smooth
terrain and we want to take larger steps.

The accumulated squared gradient in the denominator keeps increasing over time,
which continuously decreases the effective learning rate for each parameter.
As training progresses, this can lead to an excessively small learning rate,
causing the model to stop learning prematurely. This is particularly problematic
in long training sessions and for deep learning where learning rates need to be
more dynamic throughout the training process.

### RMSProp
Root Mean Squared (SMS) Propagation is an adaptive rate optimization which was
designed to address some of the shortcomings of AdaGrad. In particular it
addresses the issue of the learning rate becoming too small over time. RMSprop
modifies AdaGrad to make it more suitable for deep neural networks.

Instead of accumulating all past squared gradients, RMSprop keeps/calculates
a moving average which is used to keep track of the recent history of the
squared gradients. Recent is the key here so that we don't take into account all
the previous squared gradients, so we focus on more recent trends.
```
E[g²]ₜ= β * E[g²]ₜ₋₁ + (1 - β) * gₜ²

E[g²]ₜ= the moving average of the squared gradients for the current step.
β = beta is the decay factor and is a hyperparameter (normally something like
    0.9) and this determines how much of the past information we want to retain.
E[g²]ₜ₋₁ = the moving average of the squared gradients for the previous step.
gₜ = the current gradient.
gₜ² = the current gradient squared.
```
Lets break that down a little:
```
β * E[g²]ₜ₋₁
```
This is what carries forward the previous moving average and this is like the
memory of the algorithm since it retains information from the past step.
```
(1 - β) * gₜ²
```
This adds a portion of the current gradient squared to the moving average and is
the new information to be incorporated into the moving gradient.
So by adjusting the value of beta we can control how much of the past
information to be retained. A larger beta means that we retain more, longer
memory, of the past information and a smaller beta means that we focus on more
recent data.

So imagine that we have a parameter that at one point had a large gradient but
more recently has had smaller gradients. The moving average will be a
might eventually ignore the previous large gradient and focus on the more recent
making the gradients smoother.

The update equation then looks like this:
```
              η
θₜ₊₁ = θₜ - ----------- * gₜ
            √E[g²]ₜ + ε

θₜ₊₁ = the new updated parameter.
θₜ = the parameter for the current step.
η = eta is the global learning rate and is also a hyperparameter.
E[g²]ₜ = the moving average of the squared gradients for the current step.
ε = epsilon is a small value added to the denominator to avoid division by zero.
gₜ = the gradient for the current step.
```


### Adam
Adaptive Movement Estimation (Adam) is an adaptive learning rate optimization
algorithm that's been designed specifically for training deep neural networks.
It takes ideas from both RMSProp and Momentum.

The words that are not used frequently will have a gradient of zero.

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

### Newton's Method
The following is the normal gradient decent we saw earlier, but instead of using
θ we are using X (just to get used to using other symbols as I've see X used
frequently in places):
```
Gradient decent:
Xₖ₊₁ = Xₖ - α * ∇J(Xₖ)

Newtons method:
Xₖ₊₁ = Xₖ - [replace with matrix] * ∇J(Xₖ)
Xₖ₊₁ = Xₖ - ∇²f(Xₖ)⁻¹ * ∇J(Xₖ)
              ↑           ↑
            matrix       vector
```
The matrix is the inverse of the Hessian matrix and the vector is the gradient
of the cost function. The Hessian matrix is the second derivative of the cost
function. The Hessian matrix is a square matrix and is used to calculate the
local curvature of a function of many variables.

The result of the matrix vector multiplication is a vector that indicates how
much and in what direction to adjust Xₖ.

