def f(x):
    return x ** 2

def grad_f(x):
    return 2 * x

def sgd_momentum(x_start, learning_rate, momentum_factor, epochs):
    # x is our initial guess for the minium of the function.
    x = x_start
    # iniitially we don't have any previous momentum.
    velocity = 0

    for i in range(epochs):
        # find the gradient for x
        grad = grad_f(x)
        # momentum_factor * velocity gives us the contribution from the
        # previous velocity. And we want to move in the oposite direction of
        # the gradient to minimize the function.
        velocity = momentum_factor * velocity - learning_rate * grad
        print(f"epoch {i}: x = {x}, f(x) = {f(x)}, grad_f(x) = {grad}, velocity: {velocity}")
        x += velocity

    return x

# Parameters
x_start = 10.0
learning_rate = 0.1
momentum_factor = 0.9
epochs = 10

# Run SGD with Momentum
sgd_momentum(x_start, learning_rate, momentum_factor, epochs)

