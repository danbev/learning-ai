import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

"""
This example is a very basic linear regression example.

The known values are the xs and ys. That is, for each x value we know the
expected y value.
                              
Input Layer      Layer 1          Output Layer
   X --------->  y = mx + b  ---> Y_guess <---------> Y_actual
                                        
The first time we run this the values of m and b are just guesses, for example:

   -1.0 --->  y = 10*(-1.0) + 10  ---> 0  <------> -3.0
    0.0 --->  y = 10*( 0.0) + 10  ---> 10 <------> -1.0
    1.0 --->  y = 10*( 1.0) + 10  ---> 20 <------>  1.0
    2.0 --->  y = 10*( 2.0) + 10  ---> 30 <------>  3.0
    3.0 --->  y = 10*( 3.0) + 10  ---> 40 <------>  5.0
    4.0 --->  y = 10*( 4.0) + 10  ---> 50 <------>  7.0

The loss function is what determines the calculated y values with the know
y values. And depending on the values returned from the loss function the
parameters to the function, called the optimizer, in the layer are adjusted.
The this repeates until the values from the loss function are close to the
expected know values.
"""

print(f'TensorFlow version: {tf.__version__}')

# These are the layers in a neural network, in this case only one neuron.
# https://keras.io/api/layers/core_layers/dense
# Dense means that all the neurons are connected to every neuron in the next
# layer.
# Here we specify only one (dense) layer using the units parameter.
# The input_shape is the first layer, the input layer of our neural network and
# in our case it is just a single value (x).
dense = Dense(units=1, input_shape=[1])
model = Sequential([dense])

# Stochastic Gradient Decent (sdg)
model.compile(optimizer='sgd', loss='mean_squared_error')

# These are the x values
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
# These are the y values
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# Plot the points to get a visual of them
#plt.scatter(xs, ys)
#plt.axhline(y=np.nanmean(ys))
#plt.show()

# Run the training 500 times using the know values.
model.fit(xs, ys, epochs=500)

print("\nTraining completed.\n")
weights = dense.get_weights()
print(f'y = mx + b')
print(f'y = {weights[0][0]}x + {weights[1]}\n')
print("Expected:")
print("y = 2x + 1\n")

x = 10.0
print(f'Predict y value for {x}')
y = model.predict([x])
print(f'Predicted y to be {y}, when x is {x}')
# y = 2*10 -1
# y = 19
