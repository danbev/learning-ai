from safetensors import safe_open
import numpy as np

# Load the tensor
tensors = [
    "vision_model.gated_positional_embedding.tile_embedding.weight",
    "vision_model.gated_positional_embedding.tile_embedding.weight",
    "vision_model.gated_positional_embedding.gate",
    "vision_model.gated_positional_embedding.embedding",
    "vision_model.post_tile_positional_embedding.embedding.weight",
    "vision_model.post_tile_positional_embedding.gate",
    "vision_model.pre_tile_positional_embedding.embedding.weight",
    "vision_model.pre_tile_positional_embedding.gate",
    "vision_model.class_embedding",
    "vision_model.patch_embedding.weight",
]

with safe_open("/home/danbev/work/ai/llama-models/Llama-3.2-11B-Vision-Instruct/model-00001-of-00005.safetensors", framework="pt") as f:
#with safe_open("/home/danbev/Downloads/model-00001-of-00005.safetensors", framework="pt") as f:
    # Get the tensor

    for tensor_name in tensors:
        tensor = f.get_tensor(tensor_name)

        # Print tensor info
        print(f"\nTensor Information:")
        print(f"Name: {tensor_name}")
        print(f"Shape: {tensor.shape}")
        print(f"Type: {tensor.dtype}")
        print(f"First 10 values:")
        flattened = tensor.flatten()
        for i, val in enumerate(flattened[:50]):
            print(f"[{i}] = {val}")
