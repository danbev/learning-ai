# Multilayer Perceptron
import numpy as np
from tensorflow.keras import datasets, utils, layers, models

(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

NUM_CLASSES = 10 # categories

x_train = x_train.astype('float32') / 255.0
print(f'x_train.shape: {x_train.shape}')
x_test = x_test.astype('float32') / 255.0

y_train = utils.to_categorical(y_train, NUM_CLASSES)
print(f'y_train.shape: {y_train.shape}')
y_test = utils.to_categorical(y_test, NUM_CLASSES)


print(x_train[54, 12, 13, 1])

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
