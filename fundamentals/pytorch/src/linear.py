import torch.nn as nn
import torch

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
