import math
import numpy as np
import matplotlib.pyplot as plt

"""
Automatic gradient (autograd) calculation example from:
https://www.youtube.com/watch?v=VMj-3S1tku0&t=577s

This might diverge from the above example and contains comments about
python in addition to the concepts of automatic gradient calculation.
"""

def f(x):
    return 3*x**2 - 4*x + 5

# The derivitive of the above function is:
#  d
# --- = (3x² - 4x + 5) = 6x - 4
#  dx

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

# Plot the function
plt.figure()
plt.plot(xs, ys)
#plt.show()

# h in this case is the "nudge". It is the small increment that we nudge x to
# the right.
h = 0.0000001
x = 3.0
print(f'{f_prime(x)=}')
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


def manual_derivative_exploration():
    print('------ Manual exploration of the derivatives  ------')
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
        self._prev = set(children)
        self._op = op
        self._backward = lambda: None # recall that leaf nodes/value nodes do not have a backward function

        self.data = data
        self.label = label
        self.grad = 0.0 # does not influence the output value intially

    def __repr__(self):
        return f'Value(label={self.label}, data={self.data}, grad={self.grad}'

    def __add__(self, other):
        # This looked unfamiliar to me at first, but it is just checking if
        # other is a Value object and if not it creates one.
        # Like this would be equivalent to:
        # if isinstance(other, Value):
        #     other = other
        # else:
        #     other = Value(other)
        other = other if isinstance(other, Value) else Value(other)
        # Notice that we are returning a new Value object here, and in the
        # process documenting which objects were used to create this new
        # object, and also the operation which in this case is add.
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            # For addition the gradient of is just copied through
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __sub__(self, other):
        return self + (-other)

    def __rmul__(self, other):
        return self * other

    def __mul__(self, other):
        # Notice that we are returning a new Value object here, and in the
        # process documenting which objects were used to create this new
        # object, and also the operation which in this case is mul.
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), 'only supporting int/float powers for now'
        out =  Value(self.data**other, (self,), f'**{other}')
        def _backward():
            self.grad += other * (self.data**(other - 1)) * out.grad
        out._backward = _backward
        return out

    def __truediv__(self, other):
        return self * other**-1

    def exp(self, other):
        return Value(self.data**other.data, (self, other), '**')

    def tanh(self):
        # tanh(x) = (e^(2*x) - 1) / (e^x(2*x) + 1
        x = self.data
        tanh = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(tanh, (self,), 'tanh')
        def _backward():
            self.grad += (1 - tanh**2) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

        #print('\n'.join(map(str, topo)))

print('------ Manual exploration of the derivatives using Value object  ------')
a = Value(2.0, label='a')
b = Value(3.0, label='b')
a / b
print(f'{a=}')

b = Value(-3.0, label='b')
print(f'{b=}')

c = Value(10.0, label='c')
print(f'{c=}')

e = a*b
e.label = 'e'
print(f'{e=} (a * b)')

d = e + c
d.label = 'd'
print(f'{d=} (e + c)')

f = Value(-2.0, label='f')
print(f'{f=}')

L = d*f; L.label = 'L'
print(f'{L=}')

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
        dot.node(name = uid, label=f'Value(label={n.label}, data={n.data:.4f}, grad={n.grad:.4f})', shape='record')
        if n._op:
            dot.node(name = uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)

    for a, b in edges:
        dot.edge(str(id(a)), str(id(b)) + b._op)
    return dot

print('------ Manually calculate the derivative of the node graph  ------')
print('We apply the chain rule starting from the right most node L')
# (f(L+h) - f(L))/h
# h/h = 1.0
L.grad = 1.0
f.grad = 4.0
d.grad = -2.0

# L = d*f
# dL/dd = ?
# (f(x+h) - f(x))/h
# ((d+h)*f - d*f)/h
# ((d*f + h*f - d*f)/h
#  h*f
#  ---- = f
#   h

# d = 4.0, f = -2.0
# ((4.0+0.001)*-2.0 - 4.0*-2.0)/0.001

# Now we want to compute the derivative of L with respect to c.
# dd /dc = ?
# We know that D was created by adding c to e.
# (f(x+h) - f(x))/h
# ((c+h) + e) - (c + e)/h 
# ((c+h) + e) - 1(c + e)/h 
# (c + h + e - c - e)/h 
# h/h = 1.0

# Likewise dd/de = 1.0
# We know the value of dL/dd = -2.0 
# And know the value of dd/dc = 1.0
# Then the value of dL/dc = -2.0 * 1.0 = -2.0
c.grad = -2.0
e.grad = -2.0

# Next we want to compute dL/da. So we want to compute the derivative of a with
# respect to L. Looking at a which is called a local node its connection/link
# to L is through e which was created by multiplying a and b. 
# dL/da = (dL/de) * (de/da) = -2.0 * -3.0 = 6.0
# And de/da = b = -3.0

a.grad = -2.0 * -3.0

# And then we have dL/db. So we want to compute the derivative of b with respec
# to L:
# (f(x+h) - f(x))/h
# ((b+h) + e) - (b + e)/h 
# ((b+h) + e) - 1(b + e)/h 
# (b + h + e - b - e)/h      // b-b = 0, e-e = 0
# h/h = 1
# 1 * a = 1 * 2.0 = 2.0
# dL/db = (dL/de) * (de/db) = -2.0 * 2.0 = -4.0
b.grad = -2.0 * 2.0

digraph = draw_dot(L)
digraph.render('images/autograd', view=False, format='svg')
# The generated file can then be opened using:
# $ python -mwebbrowser autograd.svg

# Notice that what we are doing is that we are going backwards through the nodes
# and locally applying the chain rule. So we want to compute the derivative 
# with respect to L and the nodes know there children nodes, that is if the
# node was created through a computation like addition, multiplication, etc.
# We calculate the local deriviative and multiply them with the derivative of
# the parent node. This is called backpropagation.
# This is a recursive application of the chain rule backwards through the graph.

# So to recap this, we have a graph of nodes and edges. This far we have the
# following nodes: a, b, c, d, e, f, L. And we have the following edges:
# a -> e, b -> e, e -> d, c -> d, d -> L, f -> L.
# The forward pass is what creates the values for e and d and L. Initially these
# nodes only have a value. The backward pass is what computes the gradients for
# each node. We will later see that we can also have weights and biases in the
# graph. 

def manual_checking():
    print('------ Manual verifying of the derivatives  ------')
    h = 0.0000001
    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')
    e = a*b
    e.label = 'e'
    d = e + c
    d.label = 'd'
    f = Value(-2.0, label='f')
    L = d*f
    L.label = 'L'
    L1 = L.data

    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')
    e = a*b; e.label = 'e'
    d = e + c; d.label = 'd'
    f = Value(-2.0, label='f')
    L = d*f; L.label = 'L'
    L2 = L.data + h
    print(f'Derivative of L with respect to L dL/dL: {(L2-L1)/h=}')

    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')
    e = a*b; e.label = 'e'
    d = e + c; d.label = 'd'
    f = Value(-2.0 + h, label='f')
    L = d*f; L.label = 'L'
    L2 = L.data
    print(f'Derivative of L with respect to f dL/df: {(L2-L1)/h=}')

    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')
    e = a*b; e.label = 'e'
    d = e + c; d.label = 'd'
    d.data += h
    f = Value(-2.0, label='f')
    L = d*f; L.label = 'L'
    L2 = L.data
    print(f'Derivative of L with respect to d dL/dd: {(L2-L1)/h=}')

    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')
    c.data += h
    e = a*b; e.label = 'e'
    d = e + c; d.label = 'd'
    f = Value(-2.0, label='f')
    L = d*f; L.label = 'L'
    L2 = L.data
    print(f'Derivative of L with respect to c dL/dc: {(L2-L1)/h=}')

    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')
    e = a*b; e.label = 'e'
    e.data += h
    d = e + c; d.label = 'd'
    f = Value(-2.0, label='f')
    L = d*f; L.label = 'L'
    L2 = L.data
    print(f'Derivative of L with respect to e dL/de: {(L2-L1)/h=}')

    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    b.data += h
    c = Value(10.0, label='c')
    e = a*b; e.label = 'e'
    d = e + c; d.label = 'd'
    f = Value(-2.0, label='f')
    L = d*f; L.label = 'L'
    L2 = L.data
    print(f'Derivative of L with respect to b dL/db: {(L2-L1)/h=}')

    a = Value(2.0, label='a')
    a.data += h
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')
    e = a*b; e.label = 'e'
    d = e + c; d.label = 'd'
    f = Value(-2.0, label='f')
    L = d*f; L.label = 'L'
    L2 = L.data
    print(f'Derivative of L with respect to a dL/da: {(L2-L1)/h=}')

manual_checking()


print("------ Backpropagation ------")
# What we want to do is nudge our inputs to make L increase. Which we do for all
# the leaf nodes (the ones created using the Value class).
step = 0.01
print(f'Nudge leaf nodes a, b, c, and f by {step=}')
a.data += step * a.grad
b.data += step * b.grad
c.data += step * c.grad
f.data += step * f.grad

print(f'Current value of L: {L}')
print(f'Perform the forward pass which computes new values for the nodes ')
print(f'that are not leaf nodes, i.e. e, d, and L')
e = a * b
d = e + c
L = d*f

# Notice that L has increased, from -8.0 to -7.286. The value will depend on the
# step size. If we increase the step size, the value of L will increase more.
print(f'{L=}')

print("------ Neural Network Example ------")

# tahn activation function
plt.figure()
plt.plot(np.arange(-5, 5, 0.2), np.tanh(np.arange(-5, 5, 0.2)))
plt.grid()
#plt.show()

# inputs
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')
print(f'{x1=}, {x2=}')
# weights
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')
print(f'{w1=}, {w2=}')
# bias
#b = Value(6.7, label='b')
#b = Value(9.0, label='b')
b = Value(6.8813735870195432, label='b')
print(f'{b=}')
# This is the edge to the 'x1w1' node
x1w1 = x1 * w1
x1w1.label = 'x1w1'
print(f'{x1w1=}')
# This is the edge to the 'x2w2' node
x2w2 = x2 * w2
x2w2.label = 'x2w2'
print(f'{x2w2=}')

# This is the node and recall that the dot product is the sum of the products
# of the elements of the vectors.
# [x1, x2] . [w1, w2] = (x1 * w1) + (x2 * w2)
x1w1x2w2 = x1w1 + x2w2 # this is just the sum part of the "dot" product.
x1w1x2w2.label = 'x1w1x + 2w2'
print(f'{x1w1x2w2=}')

# Network but wihout the activation function.
n = x1w1x2w2 + b
n.label = 'n'
print(f'{n=}')

output = n.tanh() # this sould be n.tahn() but for that we need more funtions on the Value class
# namly exponential and division.
output.label = 'output'
print(f'{output=}')
digraph = draw_dot(output)
digraph.render('images/autograd_nn', view=False, format='svg')

# Alright, now we are doing to do the backpropagation.
#print("------ Neural Network Manual Backpropagation ------")
#output.grad = 1.0 # dL/dL = 1.0
#print(f'{output.grad=}')
## The next now in the graph, going backwards is the activation function node
## which used tanh which we need to derive.
## output = tanh(n)
## do/dn = 1 - tanh(n)^2   # and we alreay have tanh(n) which is output.data.
#print(f'do/dn = 1 - tanh(n)^2 = {1 - output.data**2}')
#n.grad = 1 - output.data**2
#
## Recall that an addtion node simply passes the gradient to all its inputs. So
## we can set the b node grad to 0.5, as well as the x1w1x2w2 node to 0.5.
#b.grad = 0.5
#x1w1x2w2.grad = 0.5
## And the moving backwards again we have another addition node, so we can set
## the x1w1 and x2w2 nodes to 0.5.
#x1w1.grad = 0.5
#x2w2.grad = 0.5
#
#x1.grad = w1.data * x1w1.grad
#w1.grad = x1.data * x1w1.grad
#
#x2.grad = w2.data * x2w2.grad
#w2.grad = x2.data * x2w2.grad
#
#digraph = draw_dot(output)
#digraph.render('autograd_nn', view=False, format='svg')

#print("------ Neural Network Manual Backpropagation ------")
#output.grad = 1.0 # this is needed as if it is 0 it will not work.
#output._backward()
#n._backward()
#b._backward()
#x1w1x2w2._backward()
#x1w1._backward()
#x2w2._backward()
#
#digraph = draw_dot(output)
#digraph.render('autograd_nn', view=False, format='svg')

print("------ Neural Network Auto Backpropagation ------")
output.backward()

# The following has now been extracted into Value.backward()
#output.grad = 1.0 # this is needed as if it is 0 it will not work.
#topo = []
#visited = set()
#def build_topo(v):
#    if v not in visited:
#        visited.add(v)
#        for child in v._prev:
#            build_topo(child)
#        topo.append(v)
#
#build_topo(output)
#print('\n'.join(map(str, topo)))
#
#for node in reversed(topo):
#    node._backward()

#output._backward()
#n._backward()
#b._backward()
#x1w1x2w2._backward()
#x1w1._backward()
#x2w2._backward()

digraph = draw_dot(output)
digraph.render('images/autograd_nn', view=False, format='svg')

import torch
x1 = torch.Tensor([2.0]).double(); x1.requires_grad = True
x2 = torch.Tensor([0.0]).double(); x2.requires_grad = True
w1 = torch.Tensor([-3.0]).double(); w1.requires_grad = True
w2 = torch.Tensor([1.0]).double(); w2.requires_grad = True
b = torch.Tensor([6.8813735870195432]).double(); b.requires_grad = True
n = x1 * w1 + x2 * w2 + b
o = torch.tanh(n)
o.backward()
print(f'{o.data.item()=}')
print(f'{x1.grad.item()=}')
print(f'{x2.grad.item()=}')
print(f'{w1.grad.item()=}')
print(f'{w2.grad.item()=}')

import random

class Neuron: 
    def __init__(self, nr_inputs):
        random.seed(18)
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nr_inputs)]
        self.b = Value(random.uniform(-1, 1))
        

    def __call__(self, x):
        #print(f'Neuron (w * x + b), w={[w.data for w in self.w]}, {x=}, {self.b.data=}')
        #print(list(zip(self.w, x)))
        #act = sum((wi*xi for wi,xi in zip(self.w, x)), Value(0.0)) + self.b
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out

    def parameters(self):
        # add the weights list to a list of the bias which is just a float.
        return self.w + [self.b]

x = [2.0, 3.0]
n = Neuron(len(x))
o = n(x)
#print(o)

class Layer:
    def __init__(self, nr_inputs, nr_neurons):
        self.neurons = [Neuron(nr_inputs) for _ in range(nr_neurons)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        # What we want is for this method to return all the parameters in all
        # the nauurons in this layer. So we got through all the neusonrs add
        # colllect the paramters into a new list which is returned.
        return [p for n in self.neurons for p in n.parameters()]

x = [2.0, 3.0]
n = Layer(2, 3)
o = n(x)
#print(o)

print("------ Multi Layer Perceptron ------")
# Multi Layer Perceptron
class MLP:
    def __init__(self, nr_inputs, nr_outputs):
        sz = [nr_inputs] + nr_outputs
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nr_outputs))]
        
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        # And like in the Layer we want to be able to collect all the
        # parameters from all the Layers.
        #return [p for layer in self.layers for lp in layer.parameters()]
        #return [p for layer in self.layers for p in layer.parameters()]
        params = []
        for layer in self.layers:
            for p in layer.parameters():
                params.append(p)
        return params

x = [2.0, 3.0, -10]
mlp = MLP(3, [4, 4, 1])
mlp_output = mlp(x)
print(mlp_output)

digraph = draw_dot(mlp_output)
digraph.render('images/autograd_mlp', view=False, format='svg')

xs = [
    [2.0, 3.0, -1.0], #input 0
    [3.0, -1.0, 0.5], #input 1
    [0.5, 1.0, 1.0],  #input 2
    [1.0, 1.0, -1.0]  #input 3
]
# * below is the unpacking operator in python
print("Input values:");
print(*xs, sep='\n')
ys = [1.0, -1.0, -1.0, 1.0] # desired targets
print("Known/true y target values:");
print(*ys, sep='\n')

print("Expected inputs with target (true/known) y values:");
for i, (x, y) in enumerate(zip(xs, ys)):
    print(f'{i} {x=}: {y=}')

print("Predicted values vs true:");
y_pred = [mlp(x) for x in xs]
for i, (p, t) in enumerate(zip(y_pred, ys)):
    print(f'{i} pred: {p.data}: true: {y}')

#print([(y_out - y_true)**2 for y_true, y_out in zip(ys, y_pred)])
#loss = sum((y_out - y_true)**2 for y_true, y_out in zip(ys, y_pred))
loss = 0.0
for i, (t, p) in enumerate(zip(ys, y_pred)):
    diff = t - p.data
    loss += diff**2
    print(f'{i} true value - predicted {diff=}, squared: {diff**2}')
print(f'loss: {loss} (the sum of the "squared" values above)')
print(''' So for each point we are going to calculate the difference between the true
 value and the predicted value. But we want to take all the values into accout
 and then get a single value. So we need to sum all the differences and we take
 the square of each difference to avoid negative values. We could also have
 taken the absolute value of each difference.''')

#loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, y_pred))
# This is the same as the loop above, just me trying to get more familiar with
# python syntax.
loss = sum(((yout - ygt)**2 for ygt, yout in zip(ys, y_pred)), Value(0.0))
# The larger the loss the further away the predicted values are from the true
# values. We want to minimize the loss.
print(f'loss: {loss.data}')

print(''' How do we minimize the loss?
 Initially we initialize the weights and biases randomly which is why these
 values are so far off. We want to adjust each of the weights and biases so
 to that the loss goes down.
 Lets take an example like the first input [2.0, 3.0, -1.0] and the first
 0 pred: -0.6168217796407953: true: 1.0.
 We need to take a look a the weights and biases for this entry.
''')
print(f'{mlp.layers[0].neurons[0].w[0]=}')
# This is a negative value. Remember that we want to change this value in way
# such that the loss goes down. If we look at the value of the loss for this
# entry we see that it is 1.6168217796407953. If we increase this value the
# loss will go up, but if we decrease this value the loss will go down.

mlp.layers[0].neurons[0].w[0] -= 0.001
print(mlp_output)
print("Predicted values after nudging weigh[0]:");
y_pred = [mlp(x) for x in xs]
for i, (p, t) in enumerate(zip(y_pred, ys)):
    print(f'{i} pred: {p.data}: true: {t}')

loss = sum(((yout - ygt)**2 for ygt, yout in zip(ys, y_pred)), Value(0.0))
# The larger the loss the further away the predicted values are from the true
# values. We want to minimize the loss.
print(f'new loss: {loss.data}')
# So that was changing one weight, but we need to nudge them up or down based
# on the gradient.

loss.backward()

print(*mlp.parameters(), sep='\n')
print(f'{mlp.layers[0].neurons[0].w[0].grad=}')
print(f'{mlp.layers[0].neurons[0].w[0].data=}')
for p in mlp.parameters():
    # The gradient is a vector that points in the direction of increated loss.
    #p.data += 0.001 * p.grad
    # Lets take the first entry:
    # data=-0.6384702733335573, grad=-2.3415567749427533
    # 0.001 * p.grad = -0.002341556774942753
    # -0.6384702733335573 + (- 0.002341556774942753)
    # -0.6384702733335573 - 0.002341556774942753) = -0.6408118301085001 
    # Notice that actually made the wieght more negative and increase the loss:
    # -0.6384702733335573
    # -0.6408118301085001 
    p.data += -0.01 * p.grad

print(f'{mlp.layers[0].neurons[0].w[0].data=}')

def pred(learning_rate):
    # forward pass through the network calling all the neurons.
    y_pred = [mlp(x) for x in xs]
    # Calculate the loss after the forward pass.
    loss = sum(((yout - ygt)**2 for ygt, yout in zip(ys, y_pred)), Value(0.0))
    print(f'new loss: {loss.data}')

    # Reset the gradients to zero.
    for p in mlp.parameters():
        # we update the gradients in place so we need to reset them to zero.
        p.grad = 0.0

    # Perform the backward pass to calculate the gradients.
    loss.backward()

    # Adjust the weights and biases with respect to the gradients.
    for p in mlp.parameters():
        p.data += -learning_rate * p.grad

    for i, (p, t) in enumerate(zip(y_pred, ys)):
        print(f'{i} pred: {p.data}: true: {t}')

for i in range(1000):
    pred(0.1)

print("The final parameters that we have learned:");
print(*mlp.parameters(), sep='\n')
