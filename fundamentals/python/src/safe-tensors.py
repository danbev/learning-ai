import torch

from safetensors.torch import save_file

tensors = {
    "embedding": torch.zeros((2, 2)),
    "attention": torch.zeros((2, 3))
}
save_file(tensors, "model.safetensors")


from safetensors import safe_open

tensors = {}
with safe_open("model.safetensors", framework="pt") as f:
    for k in f.keys():
        print(f"key: {k}")
        tensors[k] = f.get_tensor(k)

print(tensors)

tensors = {}
with safe_open("model.safetensors", framework="pt") as f:
    tensor_slice = f.get_slice("embedding")
    vocab_size, hidden_dim = tensor_slice.get_shape()
    print(f"vocab_size: {vocab_size}, hidden_dim: {hidden_dim}")
    tensor = tensor_slice[:, :hidden_dim] # change the hidden_dim to load part of the tensor


from safetensors.torch import load_model, save_model
