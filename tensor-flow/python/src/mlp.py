# Multilayer Perceptron
import numpy as np
from tensorflow.keras import datasets, utils, layers, models

(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
"""
x_train (training data) will be tensor (recall that a matrix is also a type of
tensor which has a rank/dimension of 2) with the shape:

  [50000, 32, 32, 3]

50000 is the number of images, the next two values are the pixels which is
a 32x32, and the 3 is the channel (red 0, green 1, or blue 2).
"""
print(f'x_train.shape: {x_train.shape}')
"""
y_train (training_data) is the know good values for each of the x_train values
and its shape will be:

   [50000, 1]
"""
print(f'y_train.shape: {y_train.shape}')

print(f'x_test.shape: {x_test.shape}')
print(f'y_test.shape: {y_test.shape}')

NUM_CLASSES = 10 # categories

# Scale each image so that the values are between 0 and 1.
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# One-hot encode the labels 
y_train = utils.to_categorical(y_train, NUM_CLASSES)
print(f'y_train.shape: {y_train.shape}')
y_test = utils.to_categorical(y_test, NUM_CLASSES)
print(f'y_test.shape: {y_test.shape}')


print(f'index image 54, pixel 12:13, and the green channel: {x_train[54, 12, 13, 1]}')

model = models.Sequential([
    layers.Flatten(input_shape=(32, 32, 3)),
    layers.Dense(200, activation = 'relu'),
    layers.Dense(150, activation = 'relu'),
    layers.Dense(10, activation = 'softmax'),
])

input_layer = layers.Input(shape=(32, 32, 3))
x = layers.Flatten()(input_layer)
print(f'input layer shape: {x.shape}')
x = layers.Dense(units=200, activation = 'relu')(x)
print(f'hidden layer 0 shape: {x.shape}')
x = layers.Dense(units=150, activation = 'relu')(x)
print(f'hidden layer 1 shape: {x.shape}')
output_layer = layers.Dense(units=10, activation = 'softmax')(x)
print(f'output layer shape: {output_layer.shape}')
model = models.Model(input_layer, output_layer)
