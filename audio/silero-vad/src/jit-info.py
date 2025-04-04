import torch
import pickle
import os

# Load the entire model from the .jit file
model_path = "/home/danbev/work/ai/audio/silero-vad/src/silero_vad/data/silero_vad.jit"
# Loading the model using torch.jit.load will handle all the files in the
# jit archive and reconstruct the model.
model = torch.jit.load(model_path)

# Print model structure (information for the code file)
print(model)

# Print the graph
#print(model.graph)

# Print parameters (information from the data file)
for name, param in model.named_parameters():
    print(f"Parameter: {name}, Shape: {param.shape}")

# Extract and print some statistics about each parameter
for name, param in model.named_parameters():
    tensor = param.detach()
    print(f"\nParameter: {name}")
    print(f"Shape: {tensor.shape}")
    print(f"Data type: {tensor.dtype}")
    print(f"Min value: {tensor.min().item()}")
    print(f"Max value: {tensor.max().item()}")
    print(f"Mean value: {tensor.mean().item()}")
    print(f"Standard deviation: {tensor.std().item()}")

    # Optionally, print a small sample of actual values
    if len(tensor.shape) > 1:
        flat_tensor = tensor.flatten()
        print(f"Sample values: {flat_tensor[:5].tolist()}")
    else:
        print(f"Sample values: {tensor[:5].tolist()}")
