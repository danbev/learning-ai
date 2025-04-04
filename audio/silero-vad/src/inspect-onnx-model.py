import onnx
import onnxruntime as ort
import numpy as np

# Load the ONNX model
model_path = "/home/danbev/work/ai/audio/silero-vad/src/silero_vad/data/silero_vad.onnx"
model = onnx.load(model_path)

# Basic model validation
onnx.checker.check_model(model)

# Print model metadata
print(f"Model IR version: {model.ir_version}")
print(f"Producer name: {model.producer_name}")
print(f"Producer version: {model.producer_version}")
print(f"Domain: {model.domain}")
print(f"Model version: {model.model_version}")
print(f"Doc string: {model.doc_string}")

# Get input and output details
print("\nInputs:")
for input in model.graph.input:
    print(f"  {input.name}: {input.type.tensor_type.elem_type}")
    # Print shape information if available
    if input.type.tensor_type.shape:
        dims = []
        for dim in input.type.tensor_type.shape.dim:
            if dim.dim_param:
                dims.append(dim.dim_param)
            else:
                dims.append(dim.dim_value)
        print(f"  Shape: {dims}")

print("\nOutputs:")
for output in model.graph.output:
    print(f"  {output.name}: {output.type.tensor_type.elem_type}")
    # Print shape information if available
    if output.type.tensor_type.shape:
        dims = []
        for dim in output.type.tensor_type.shape.dim:
            if dim.dim_param:
                dims.append(dim.dim_param)
            else:
                dims.append(dim.dim_value)
        print(f"  Shape: {dims}")

# Count nodes and operators
op_types = [node.op_type for node in model.graph.node]
op_count = {}
for op in op_types:
    if op in op_count:
        op_count[op] += 1
    else:
        op_count[op] = 1

print("\nOperators used:")
for op, count in op_count.items():
    print(f"  {op}: {count} instances")

# Test inference with proper inputs for all required parameters
session = ort.InferenceSession(model_path)

# Prepare all inputs
input_feed = {}

# Create input with proper shape - the error shows we need rank 2 for input
# For audio features, this is likely [batch_size, sequence_length]
input_feed["input"] = np.random.randn(1, 512).astype(np.float32)

# Create state with proper shape - error shows we need rank 3, not 4
# The shape should be [2, batch_size, hidden_size]
input_feed["state"] = np.zeros((2, 1, 128), dtype=np.float32)

# Create sample rate input - this is a scalar value
input_feed["sr"] = np.array(16000, dtype=np.int64)

# Print shapes for verification
for name, tensor in input_feed.items():
    print(f"Created dummy {name} with shape: {tensor.shape}")

# Run inference
output_names = [output.name for output in session.get_outputs()]
outputs = session.run(output_names, input_feed)

# Print output shapes
for i, output_name in enumerate(output_names):
    print(f"Output '{output_name}' shape: {outputs[i].shape}")
