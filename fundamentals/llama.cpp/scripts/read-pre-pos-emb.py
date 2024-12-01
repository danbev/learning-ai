#!/usr/bin/env python3
import logging
import sys
from pathlib import Path
from gguf.gguf_reader import GGUFReader
import numpy as np

logger = logging.getLogger("reader")

def read_position_embeddings(gguf_file_path, tensor_name, row_idx=6, num_values=10):
    """
    Reads and prints information about position embeddings from a GGUF file.

    Parameters:
    - gguf_file_path: Path to the GGUF file
    - tensor_name: Name of the tensor to inspect
    - row_idx: Which row to read (default 6)
    - num_values: How many values to print (default 10)
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
    print(f"Shape: {target_tensor.shape}")  # This should already be correct
    print(f"Type: {target_tensor.tensor_type.name}")
    print(f"Total elements: {target_tensor.n_elements}")

    # Get the tensor data
    data = target_tensor.data
    if isinstance(data, np.ndarray):
        # Print the actual shape we got
        print(f"Actual data shape: {data.shape}")
        print(f"Number of rows: {data.shape[0]}")
        print(f"Elements per row: {data.shape[1]}")

        # Get the specified row
        row_data = data[row_idx]

        print(f"\nFirst {num_values} values from row {row_idx}:")
        for i in range(num_values):
            print(f"[{i}] = {row_data[i]}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: reader.py <path_to_gguf_file> [row_index] [num_values]")
        sys.exit(1)

    gguf_file_path = sys.argv[1]
    row_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    num_values = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    read_position_embeddings(
        gguf_file_path,
        #"v.enc.pre_tile_pos_embd.weight",
        "v.pre_tile_position_embd.weight",
        row_idx,
        num_values
    )
