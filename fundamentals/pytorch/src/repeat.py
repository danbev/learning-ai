import torch

num_heads = 4                             # Total number of heads/channels in the hidden dimension
n_groups  = 2                             # Number of groups
expansion_factor = num_heads // n_groups  # = 2

B_tensor = torch.tensor([
    [
        [
            [10.0],  # G1 (Group 1 data)
            [20.0]   # G2 (Group 2 data)
        ]
    ]
])
# B_tensor shape: torch.Size([1, 1, 2, 1])

print(f"Original B Tensor (Groups G1, G2):")
print(B_tensor.squeeze())
print("");

print(f"repeat_interleave:")
# This shuffles the blocks (interleaves).
B_interleaved = B_tensor.repeat_interleave(expansion_factor, dim=2)

print(f"   Layout: [G1, G1, G2, G2]")
print(B_interleaved.squeeze())
print(f"   Shape: {B_interleaved.shape}")
print("");


print(f"repeat:")
# This duplicates the entire sequence of blocks.
B_repeated = B_tensor.repeat(1, 1, expansion_factor, 1)

print(f"   Layout: [G1, G2, G1, G2]")
print(B_repeated.squeeze())
print(f"   Shape: {B_repeated.shape}")
print("");

# The two expanded tensors have the exact same elements but in a different order.
print("Numerical Mismatch Check:")
print(f"B_interleaved != B_repeated: {not torch.allclose(B_interleaved, B_repeated)}")
