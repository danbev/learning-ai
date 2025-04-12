from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
import onnx
import torch
from torch.profiler import profile, record_function, ProfilerActivity

def print_all_submodule_code(module, prefix=''):
    print(f"{prefix}Module: {module._get_name()}")
    print(f"{prefix}Code:\n{module.code}\n")

    for name, submodule in module.named_children():
        print(f"{prefix}Submodule: {name}")
        print_all_submodule_code(submodule, prefix + '  ')

model = load_silero_vad(onnx=False)
print(model._model)

#print(model.code)
#print(model.graph)
#model.code()

#stft = model._model.stft
#print_all_submodule_code(stft)

#encoder = model._model.encoder
#print_all_submodule_code(encoder)

#decoder = model._model.decoder
#print(decoder)
#print(decoder.code)


#print_all_submodule_code(model)

# Print the model's graph
#print(model.graph)
# Print the model's graph
#print(model.code)
#state_dict = model.state_dict()

#for attr_name in dir(model):
#    if not attr_name.startswith('_'):  # Skip private attributes
#        attr = getattr(model, attr_name)
#        if not callable(attr):  # Skip methods
#            print(f"Attribute {attr_name}: {attr}")

# List all buffers (non-parameter tensors)
#for name, buffer in model.named_buffers():
#    print(f"Buffer {name}: shape={buffer.shape}")

# Run model with sample input
#model(sample_input)

#members = [attr for attr in dir(model) if not attr.startswith('__')]
#print(members)
#print(model.named_parameters)
#print(model.modules)

#model_state = model.state_dict()
#for key, tensor in model_state.items():
#    print(f"Layer: {key}")
#    print(f"  Shape: {tensor.shape}")
#    print(f"  Dtype: {tensor.dtype}")
#    # Optionally print sample values
#    print(f"  Sample: {tensor.flatten()[:3].tolist()}")
#    tensor_list = tensor.flatten()[:10].tolist()
#    print("  Sample:")
#    for element in tensor_list:
#        print(f"    {element}")

def show_module_info(module):
    print(f"Module: {module._get_name()}")
    for name, param in module.named_parameters():
        print(f"Parameter: {name}, Shape: {param.shape}")

    for name, buffer in module.named_buffers():
        print(f"Buffer: {name}, Shape: {buffer.shape}")

# List all methods available in the STFT module
    methods = [method for method in dir(module) if not method.startswith('_') or method == '__call__']
    print(f"Available methods: {methods}")

# Try to access the transform_ method directly
    try:
        transform_code = model._model.stft.transform_.code
        print(f"Transform method code:\n{transform_code}")
    except AttributeError:
        print("Couldn't access transform_ method directly")

show_module_info(model)
show_module_info(model._model.stft)
#show_module_info(model._model.encoder)
#show_module_info(model._model.decoder)

input_tensor = torch.zeros(1, 512)  # Same as your first chunk
def extract_intermediates(model, input_tensor, sr=16000):
    # Run the model
    result = model(input_tensor, sr)

    try:
        stft_module = model._model.stft
        stft_out = stft_module(input_tensor)
        print(f"STFT output shape: {stft_out.shape}");
        encoder = model._model.encoder
        encoder_out = encoder(stft_out)
        print(f"Encoder output shape: {encoder_out.shape}");
        print(encoder_out)
    except Exception as e:
        print(f"Could not extract STFT output: {e}")

    # Return the original result
    return result

# Use the function
#result = extract_intermediates(model, input_tensor)
