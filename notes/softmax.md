## Softmax
This function is used in neural networks. It takes as input a vector of real 
number:
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
