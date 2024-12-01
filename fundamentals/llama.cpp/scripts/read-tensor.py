#!/usr/bin/env python3
import logging
import sys
from pathlib import Path
from gguf.gguf_reader import GGUFReader
import numpy as np

logger = logging.getLogger("reader")

def read_tensor_values(gguf_file_path, tensor_name):
    """
    Reads and prints information about a specific tensor from a GGUF file.

    Parameters:
    - gguf_file_path: Path to the GGUF file
    - tensor_name: Name of the tensor to inspect
    """
    reader = GGUFReader(gguf_file_path)

    # Find the specified tensor
    target_tensor = None
    for tensor in reader.tensors:
        if tensor.name == tensor_name:
            target_tensor = tensor
            break

    if target_tensor is None:
        print(f"Tensor '{tensor_name}' not found in the model")
        return

    # Print tensor information
    print(f"\nTensor Information:")
    print(f"Name: {target_tensor.name}")
    print(f"Shape: {' x '.join(map(str, target_tensor.shape))}")
    print(f"Type: {target_tensor.tensor_type.name}")
    print(f"Total elements: {target_tensor.n_elements}")

    # Get the tensor data
    data = target_tensor.data
    if isinstance(data, np.ndarray):
        print("\nFirst 10 values:")
        values = data.flatten()[:100]
        for i, value in enumerate(values):
            print(f"[{i}] = {value}")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: reader.py <path_to_gguf_file> <tensor_name>")
        sys.exit(1)

    gguf_file_path = sys.argv[1]
    tensor_name = sys.argv[2]
    read_tensor_values(gguf_file_path, tensor_name)
