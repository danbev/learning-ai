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
# The derivitive of the above function is:
# 3 * x² - 4x + 5
# 3 * 2x - 4x¹
# 3 * 2x - 1*4x⁰
# 3 * 2x - 1*4*1
# 6x - 4
def f_prime(x):
    return float(6*x - 4)

print(f'{f(3.0)=}')

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

plt.plot(xs, ys)
#plt.show()

h = 0.0000001
x = 3.0
# Definition of the derivative of a function:
# f'(x) = lim h->0 (f(x+h) - f(x))/h
print(f'{h=}')
print(f'{x=}')
print(f'{f(x+h)=}')
print(f'{f(x+h) - f(x)=} is the nudge of x')
print(f'{(f(x+h) - f(x))/h=} is that slop at x=3')
print(f'{f_prime(x)=}')
# We can inspect the plot above and see that the slope will be positive when
# we nudge x to the right.
# If x is negative and we nudge, that is add to it, then the slope will be
# negative.
x = -3.0
print(f'{x=}')
print(f'{f(x+h)=:.10f}')
print(f'{f(x+h) - f(x)=:.10f} is the nudge of x')
print(f'{(f(x+h) - f(x))/h=:.10f} is that slop at x=3')
print(f'{f_prime(x)=:.10f}')

x = 2/3
print(f'{x=}')
print(f'{f(x+h)=}')
print(f'{f(x+h) - f(x)=:.15f} is the nudge of x')
print(f'{(f(x+h) - f(x))/h=:.15f} is that slop at x=3')
print(f'{f_prime(x)=}')


print('--------------------')
h = 0.0001
# inputs
a = 2.0
b = -3.0
c = 10.0

d1 = a*b + c # output
a += h
d2 = a*b + c
print('"Bumping a"')
print(f'{d1=}')
print(f'{d2=}')
print(f'slope: {(d2 - d1)/h}')

# Next, let dump b a little...
print('"Bumping b"')
a = 2.0
b = -3.0
c = 10.0

d1 = a*b + c
b += h
d2 = a*b + c
print(f'{d1=}')
print(f'{d2=}')
print(f'slope: {(d2 - d1)/h}')


# Next, let dump c a little...
print('"Bumping c"')
a = 2.0
b = -3.0
c = 10.0

d1 = a*b + c
c += h
d2 = a*b + c
print(f'{d1=}')
print(f'{d2=}')
print(f'slope: {(d2 - d1)/h}')

class Value:
    def __init__(self, data):
        self.data = data

    def __repr__(self):
        return f'Value(data={self.data})'

    def __add__(self, other):
        return Value(self.data + other.data)

    def __mul__(self, other):
        return Value(self.data * other.data)

a = Value(2.0)
print(a)
b = Value(-3.0)
print(a+b)
c = Value(10.0)
d = a*b + c
print(d)
