## Softmax
This function is used in neural networks. It takes as input a vector of real 
numbers:
```
x = [x1, x2, ..., xn]

softmax(x₁) =                exp(x₁)
               ----------------------------------
               (exp(x₁) + exp(x₂) + ... + exp(xₙ))

prop_x₁ = softmax(x₁) = exp(x₁) / (exp(x₁) + exp(x₂) + ... + exp(xₙ))
prop_x₂ = softmax(x₂) = exp(x₂) / (exp(x₁) + exp(x₂) + ... + exp(xₙ))
...
```
Notice that we need to calculate the sum of all the exponentials of the input
which is then used to divide (normalize) each exponential.


In the case of a neural network the last layer will be a matrix of logits, that
is raw unnormalized predictions. Each row in this matrix corresponds to an
entry in the input sequence. So to visualize this we can think of this matrix
as:
```
'Dan':   [0.1, 0.2, 0.3, 0.4]
'loves': [0.5, 0.6, 0.7, 0.8]
'ice':   [0.9, 1.0, 1.1, 1.2]
'cream': [1.3, 1.4, 1.5, 1.6]
```
Think of each row in the logits matrix as a set of scores given by the network,
However, these scores are not probabilities yet—they can be any real number. The
softmax function's role is to transform these scores into actual probabilities.
What we want to do is to normalize each row of this matrix. So we apply the
softmax function to each row: 
```
'Dan':   [0.1, 0.2, 0.3, 0.4] -> [0.1, 0.2, 0.3, 0.4]
'loves': [0.5, 0.6, 0.7, 0.8] -> [0.1, 0.2, 0.3, 0.4]
'ice':   [0.9, 1.0, 1.1, 1.2] -> [0.1, 0.2, 0.3, 0.4]
'cream': [1.3, 1.4, 1.5, 1.6] -> [0.1, 0.2, 0.3, 0.4]
```
The softmax function is defined as:
```
σ(x)_i =   eˣⁱ
         ------------
          ∑_j eˣʲ

σ(x)_i = represents the i-th element of the output vector
eˣⁱ = exponential of the i-th element of the input vector
∑_j eˣʲ = sum of all the exponentials of the input vector
```

### Numerical Stability
The standard softmax function involves computing the exponential of numbers. If
these numbers are very large or very small, it can lead to problems.
Exponentiating a large number can lead to overflow, where the number becomes too
large for the computer to represent accurately, resulting in infinity.
Exponentiating a very small (negative) number can lead to underflow, where the
number becomes so close to zero that it is rounded down to zero, losing accuracy.

A common technique to make softmax numerically stable is to subtract the maximum
value in the input vector from each element before exponentiation. This process
is known as softmax stabilization.
This subtraction doesn't change the output of the softmax function (due to the
properties of the exponential function and the way softmax normalizes its
output), but it prevents the input values to the exponential function from being
 too large.
As a result, the risk of overflow is significantly reduced, while the relative
differences between input values are preserved, ensuring accurate and stable
softmax calculations.
