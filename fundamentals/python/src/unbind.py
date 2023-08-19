import torch

# Create a 2D tensor with shape (3, 4)
#                        ----> dim 0
tensor = torch.tensor([[1, 2, 3, 4],          # |
                       [5, 6, 7, 8],          # | dim = 1
                       [9, 10, 11, 12]])      # v 
print(f'original tensor: {tensor}')

# Unbind the tensor along the first dimension (default behavior)
slices = tensor.unbind()
print(f'slices is of type: {type(slices)}, and has length: {len(slices)}')
# slices is a tuple of tensors
print(slices)

# Print the resulting slices
print('Slices along dimension 0:')
for slice_tensor in slices:
    print(slice_tensor)

# Unbind the tensor along the second dimension (dim=1)
slices = tensor.unbind(dim=1)

# Print the resulting slices
print('Slices along dimension 1:')
for slice_tensor in slices:
    print(slice_tensor)
