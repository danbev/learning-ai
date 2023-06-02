import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

dense = Dense(units=1, input_shape=[1])
model = Sequential([dense])

# Stochastic Gradient Decent (sdg)
model.compile(optimizer='sgd', loss='mean_squared_error')

# These are the x values
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
# These are the y values
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)
# y = 2x - 1

model.fit(xs, ys, epochs=500)

# ask the model to predict what y will be when x is 10.0
x = 10.0
y = model.predict([x])
print(f'predicted y to be {y}, when x is {x}')
weights = dense.get_weights()
print(f'y = {weights[0][0]}x {weights[1]}')
# y = 2*10 -1
# y = 19
