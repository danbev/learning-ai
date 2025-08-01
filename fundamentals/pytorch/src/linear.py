import torch.nn as nn
import torch.nn.functional as F
import torch
import inspect

import torch._C

linear = nn.Linear(3, 2)

new_weight = torch.tensor([[1.0, 2.0, 3.0],   # First output neuron weights
                          [4.0, 5.0, 6.0]])   # Second output neuron weights

new_bias = torch.tensor([10.0, 20.0])

linear.weight = nn.Parameter(new_weight) # Basically a Tensor but needs to be a Parameter to be registered as a model parameter
linear.bias = nn.Parameter(new_bias)

print(f"Parameter weight: {linear.weight}")
print()

# Assign as parameters (this preserves gradient tracking)
linear.weight = nn.Parameter(new_weight)
linear.bias = nn.Parameter(new_bias)

print("Linear layer:", linear)
#print("nn.Linear __dict__ contents:")
#for key, value in linear.__dict__.items():
#    print(f"  {key}: {type(value)} = {value}")

input_data = torch.tensor([[1.0, 2.0, 3.0],
                          [4.0, 5.0, 6.0]])
print("Input data:", input_data)

with torch.no_grad():
    output = linear.forward(input_data)

#print("Output data:", output.detach().numpy())
print("Output data:", output.numpy())

print(inspect.getsource(linear.forward))
print(inspect.getsource(F.linear))

print("torch._C is the C extension module:")
print(f"   Type: {type(torch._C)}")
print(f"   Module: {torch._C}")

if hasattr(torch._C, '_nn'):
    print(f"\ntorch._C._nn exists: {torch._C._nn}")
    print(f"   Type: {type(torch._C._nn)}")

    # Find linear-related functions
    nn_attrs = [attr for attr in dir(torch._C._nn) if 'linear' in attr.lower()]
    print(f"   Linear-related attributes: {nn_attrs}")

    if hasattr(torch._C._nn, 'linear'):
        linear_func = torch._C._nn.linear
        print(f"\ntorch._C._nn.linear:")
        print(f"   Type: {type(linear_func)}")
        print(f"   Same as F.linear? {F.linear is linear_func}")
