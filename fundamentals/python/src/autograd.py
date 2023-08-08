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
    def __init__(self, data, children=(), op='', label=''):
        self.data = data
        self._prev = set(children)
        self._op = op
        self.label = label
        self.grad = 0.0 # does not influence the output value intially

    def __repr__(self):
        return f'Value(data={self.data})'

    def __add__(self, other):
        # Notice that we are returning a new Value object here, and in the
        # process documenting which objects were used to create this new
        # object, and also the operation which in this case is add.
        return Value(self.data + other.data, (self, other), '+')

    def __mul__(self, other):
        # Notice that we are returning a new Value object here, and in the
        # process documenting which objects were used to create this new
        # object, and also the operation which in this case is mul.
        return Value(self.data * other.data, (self, other), '*')

a = Value(2.0, label='a')
print(f'{a=}')
b = Value(-3.0, label='b')
print(f'{a+b=}')
c = Value(10.0, label='c')
e = a*b; e.label = 'e'
d = e + c; d.label = 'd'
f = Value(-2.0, label='f')
L = d*f; L.label = 'L'
print(f'{d=}')
print(f'{d._prev=}')
# By using _prev we can figure out which Value objects were used to create d.
ds = list(d._prev)
# And we can go backwards to figure out how d was created.
print(f'{ds[0]=}, {ds[0]._op=}')
print(f'{ds[1]=}, {ds[1]._op=}, {ds[1]._prev=}')
print(f'{d._op=}')

from graphviz import Digraph
def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})
    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        dot.node(name = uid, label=f'Value(label={n.label}, data={n.data:.2f}, grad={n.grad})', shape='record')
        if n._op:
            dot.node(name = uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)

    for a, b in edges:
        dot.edge(str(id(a)), str(id(b)) + b._op)
    return dot

digraph = draw_dot(L)
digraph.render('autograd', view=False, format='svg')
# The generated file can then be opened using:
# $ python -mwebbrowser autograd.svg

