import numpy as np
import matplotlib.pyplot as plt

def f(x): 
    return 2 * x

def f_prime(x):
    # The derivative of f(x) = 2x is 2.
    return 2

# So we have 5 input values and 5 target values.
X = np.array([1, 2, 3, 4, 5])
# And notice that y are the target value, that is the known true output values.
y = f(X)
print(f"X (inputs): {X}")
print(f"y (targets): {y}")

# Initial weight which is what we are trying to learn, which should be
# the derivative of f(x) = 2x.
weight = 1.0
# Initial bias which is what we are trying to learn which is just 0 in this case.
bias = 0.0

learning_rate = 0.01

epochs = 1000

def plot(x, y):
    # Plot the function to visualize it
    x = np.linspace(-10, 10, 100)
    y2 = 2 * x
    plt.scatter(x[0], y[0], color='red')
    plt.scatter(x[1], y[1], color='red')
    plt.scatter(x[2], y[2], color='red')
    plt.scatter(x[3], y[3], color='red')
    plt.scatter(x[4], y[4], color='red')
    plt.plot(x, y2, label='y = 2x')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.title('Plot of the function y = 2x')
    plt.show()

#plot(X, y)

y2 = 2 * X
plt.plot(X, y2, label='y = 2x')
plt.xlim(0, 13)
plt.ylim(0, 13)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.title('Plot of the function y = 2x')
plt.xticks(np.arange(0, 13, 1))
plt.yticks(np.arange(0, 13, 1))

colors = ['red', 'green', 'blue', 'gray', 'purple', 'black', 'yellow', 'pink', 'brown', 'orange', 'magenta']

color_idx = 0
for epoch in range(epochs):
    # Recall that this will first perform scalar multiplication to each element
    # in the X array and then add the bias to each element in the array.
    y_pred = weight * X + bias

    # Loss function (Mean Squared Error)
    loss = np.mean((y_pred - y) ** 2)

    # Derivatives (gradients) with respect to weight and bias
    d_weight = 2 * np.mean((y_pred - y) * X)
    d_bias = 2 * np.mean(y_pred - y)

    # Update weight and bias using gradients
    weight -= learning_rate * d_weight
    bias -= learning_rate * d_bias

    if epoch % 100 == 0:
        plt.scatter(X[0], y_pred[0], color=colors[color_idx])
        plt.scatter(X[1], y_pred[1], color=colors[color_idx])
        plt.scatter(X[2], y_pred[2], color=colors[color_idx])
        plt.scatter(X[3], y_pred[3], color=colors[color_idx])
        plt.scatter(X[4], y_pred[4], color=colors[color_idx])
        print(f"{epoch}, Loss: {loss:.6f}, y_pred: {y_pred}, color={colors[color_idx]}")
        color_idx += 1

print(f"Final weight: {weight}, Final bias: {bias}, color={colors[color_idx]}")
print(f"Learned function: y = 2 * {weight}")
print(f"Target function: y = 2x")
plt.scatter(X[0], y_pred[0], color=colors[color_idx])
plt.scatter(X[1], y_pred[1], color=colors[color_idx])
plt.scatter(X[2], y_pred[2], color=colors[color_idx])
plt.scatter(X[3], y_pred[3], color=colors[color_idx])
plt.scatter(X[4], y_pred[4], color=colors[color_idx])

plt.show()
