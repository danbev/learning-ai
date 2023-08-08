import math
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 3*x**2 - 4*x + 5

print(f(3.0))
# 3*3² - 4*3 + 5
# 3*9 - 12 + 5
# 27 - 12 + 5
# 27 - 12 + 5
# 15 + 5 = 20

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

