## Activation Functions

### ReLU
The ReLU function is defined as:
```python
def relu(x):
    return max(0, x)
```
So if x is greater than 0 it acts like an identity function, but if x is less
than 0 it returns 0. This is a very simple function, but it is very effective
for neural networks. It is also very fast to compute. 

### Leaky ReLU
The Leaky ReLU function is defined as:
```python
def leaky_relu(x):
    return max(0.01*x, x)

```
This function is similar to the ReLU function, but it has a small slope for
x = 2,  leaky_relu(2) = 2, but leaky_relu(-2) = -0.02. This is done to avoid
the dying ReLU problem, where the gradient of the ReLU function is 0 for x < 0,
which means that the weights will not be updated during backpropagation.

### ELU
ELU stands for Exponential Linear Unit and is defined as:
```python
def elu(x):
    if x < 0:
        return alpha * (exp(x) - 1)
    else:
        return x
```


