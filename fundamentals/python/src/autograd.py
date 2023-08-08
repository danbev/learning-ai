import math
import numpy as np
import matplotlib.pyplot as plt

"""
Automatic gradient calculation example from:
https://www.youtube.com/watch?v=VMj-3S1tku0&t=577s

This might divert from the above example and contains comments about
python in addition to the concepts of automatic gradient calculation.
"""

def f(x):
    return 3*x**2 - 4*x + 5

print(f(3.0))

xs = np.arange(-5, 5, 0.25)
print(f'{xs=}')
# Notice that passing a numpy array to the function numpy will take care of
# broadcasting the function over the array. This had me a little confused at
# first.
ys = f(xs)
print(f'{ys=}')
# We would have to write the following without the broadcasting
ys = np.array([f(x) for x in xs])
print(f'{ys=}')

