import numpy as np

# X and Y are numpy arrays storing our dataset
X = np.array([1, 2, 3, 4, 5])
Y = np.array([2, 4, 5, 4, 5])

# Initialize parameters
alpha = 0.01  # learning rate
epochs = 1000  # number of iterations
m = 0  # initial guess for slope
b = 0  # initial guess for y-intercept
n = len(X)

# Gradient descent algorithm
for i in range(epochs):
    Y_pred = m*X + b  # Current predicted value of Y
    D_m = (-2/n)*sum(X*(Y-Y_pred))  # Derivative wrt m
    D_b = (-2/n)*sum(Y-Y_pred)  # Derivative wrt b
    m = m - alpha * D_m  # Update m
    b = b - alpha * D_b  # Update b

print("After {0} iterations m = {1}, b = {2}".format(epochs, m, b))

