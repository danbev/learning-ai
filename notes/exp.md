## The Exponential Function
This is used in many places in AI/ML and math in general. It is used in the
softmax function, and in the Bradley-Terry model. The exponential function is
defined as:
```
exp(x) = e^x
```
Where `e` is Euler's number, approximately equal to 2.71828. Notice that this
function will always return a positive number, and it is always increasing.

This is very use for BT where the output represents probabilities and cannot
be negative.

## Amplification of Differences
The exponential function amplifies differences between numbers. Small
differences in the input (latent rewards) become larger differences in the
output (probabilities). This characteristic is particularly useful in preference
modeling and other applications where we want to magnify the impact of
differences in quality, strength, or preference levels between options.
