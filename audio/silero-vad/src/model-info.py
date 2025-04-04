import onnx
import numpy as np
from collections import defaultdict

# Load the ONNX model
#model_path = "/home/danbev/work/ai/audio/silero-vad/src/silero_vad/data/silero_vad.onnx"
model_path = "/home/danbev/work/ai/learning-ai/audio/silero-vad/venv/lib/python3.12/site-packages/silero_vad/data/silero_vad.jit"
model = onnx.load(model_path)

# Create a dictionary to store all tensor information
tensors = defaultdict(dict)

# Get input tensors
print("=== INPUT TENSORS ===")
for input in model.graph.input:
    name = input.name
    shape = []
    if input.type.tensor_type.shape:
        for dim in input.type.tensor_type.shape.dim:
            if dim.dim_param:
                shape.append(dim.dim_param)
            else:
                shape.append(dim.dim_value)
    data_type = input.type.tensor_type.elem_type
    print(f"{name}: shape={shape}, type={data_type}")
    tensors[name]['shape'] = shape
    tensors[name]['type'] = data_type
    tensors[name]['role'] = 'input'

# Get output tensors
print("\n=== OUTPUT TENSORS ===")
for output in model.graph.output:
    name = output.name
    shape = []
    if output.type.tensor_type.shape:
        for dim in output.type.tensor_type.shape.dim:
            if dim.dim_param:
                shape.append(dim.dim_param)
            else:
                shape.append(dim.dim_value)
    data_type = output.type.tensor_type.elem_type
    print(f"{name}: shape={shape}, type={data_type}")
    tensors[name]['shape'] = shape
    tensors[name]['type'] = data_type
    tensors[name]['role'] = 'output'

# Print all nodes to find where tensors are defined
print("\n=== NODE OUTPUTS ===")
for i, node in enumerate(model.graph.node):
    for output in node.output:
        print(f"Node {i} ({node.op_type}): output tensor '{output}'")

# Recursive function to print details from subgraphs
def process_subgraph(graph, prefix=""):
    # Print node outputs in this subgraph
    print(f"\n=== SUBGRAPH {prefix} NODE OUTPUTS ===")
    for i, node in enumerate(graph.node):
        for output in node.output:
            print(f"Node {i} ({node.op_type}): output tensor '{output}'")
        
        # Check if this node has tensor attributes (constants)
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.TENSOR:
                if hasattr(attr.t, 'dims') and attr.t.dims:
                    print(f"Node {i} ({node.op_type}): tensor attribute '{attr.name}' shape={attr.t.dims}")
    
    # Print weights in this subgraph
    print(f"\n=== SUBGRAPH {prefix} WEIGHTS ===")
    for init in graph.initializer:
        shape = [dim for dim in init.dims] if hasattr(init, 'dims') else []
        if not shape and hasattr(init, 'type') and hasattr(init.type, 'tensor_type'):
            shape = [dim.dim_value for dim in init.type.tensor_type.shape.dim]
        
        # Try another approach if shape is still empty
        if not shape:
            try:
                # Get raw data and estimate shape from size
                if init.raw_data:
                    data_size = len(init.raw_data)
                    print(f"{init.name}: data_size={data_size} bytes")
                    continue
            except:
                pass
        
        data_type = init.data_type if hasattr(init, 'data_type') else 'unknown'
        print(f"{init.name}: shape={shape}, type={data_type}")
    
    # Check for subgraphs in this subgraph
    for node in graph.node:
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                sub_prefix = f"{prefix}_{node.name}_{attr.name}"
                process_subgraph(attr.g, sub_prefix)

# Process If nodes to find subgraphs
print("\n=== PROCESSING SUBGRAPHS ===")
for node in model.graph.node:
    if node.op_type == 'If':
        print(f"\nFound If node: {node.name}")
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                print(f"Processing subgraph: {attr.name}")
                process_subgraph(attr.g, f"{node.name}_{attr.name}")

# Look for constants in the main graph
print("\n=== CONSTANT TENSORS ===")
for i, node in enumerate(model.graph.node):
    if node.op_type == 'Constant':
        for attr in node.attribute:
            if attr.name == 'value':
                # Try to extract shape from the tensor
                try:
                    shape = list(attr.t.dims) if hasattr(attr.t, 'dims') else []
                    if shape:
                        print(f"Constant node {i}: output={node.output[0]}, shape={shape}")
                except:
                    print(f"Constant node {i}: output={node.output[0]}, shape=unknown")
