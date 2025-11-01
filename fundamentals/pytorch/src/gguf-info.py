import torch
from gguf import GGUFReader

def print_tensors(r: GGUFReader):
    print(f"Total number of tensors: {len(r.tensors)}\n")
    for tensor in r.tensors:
        print(f"Name: {tensor.name}")
        print(f"  shape: {tensor.shape}, type: {tensor.tensor_type}, elements: {tensor.n_elements}")

def print_tensor(r: GGUFReader, name: str):
    tensor = next(t for t in r.tensors if t.name == name)
    print(f"Tensor: {tensor.name}")
    print(f"  shape: {tensor.shape}, type: {tensor.tensor_type}, elements: {tensor.n_elements}")
    print(f"  data:\n{tensor.data}")

def print_field(r: GGUFReader, field_name: str):
    field = r.get_field(field_name)
    print(f"Field: {field_name}")
    print(f"  contents: {field.contents()}")
    print(f"  type: {type(field)}, value: {field}")

def main():
    gguf_path = "/home/danbev/work/ai/llama.cpp/models/granite-4.0-h-tiny-Q4_0.gguf"
    r = GGUFReader(gguf_path)
    #print_tensors(r)
    print_tensor(r, "blk.39.ssm_in.weight")

    print_field(r, "granitehybrid.expert_used_count")
if __name__ == "__main__":
    main()
