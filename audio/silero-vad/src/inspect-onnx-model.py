import onnx
import re

def print_onnx_model_structure(model_path):
    model = onnx.load(model_path)
    
    # Convert the model to a string representation
    model_str = str(model)
    
    # Replace raw data sections with a placeholder
    filtered_str = re.sub(r'raw_data: ".*?"', 'raw_data: "<TENSOR_DATA_FILTERED>"', model_str, flags=re.DOTALL)
    
    # Further clean up to remove any remaining large binary data
    filtered_str = re.sub(r'float_data: .*?\n', 'float_data: [<FILTERED>]\n', filtered_str)
    filtered_str = re.sub(r'int32_data: .*?\n', 'int32_data: [<FILTERED>]\n', filtered_str)
    filtered_str = re.sub(r'int64_data: .*?\n', 'int64_data: [<FILTERED>]\n', filtered_str)
    filtered_str = re.sub(r'double_data: .*?\n', 'double_data: [<FILTERED>]\n', filtered_str)
    
    print(filtered_str)

# Usage
model_path = "/home/danbev/work/ai/audio/silero-vad/src/silero_vad/data/silero_vad.onnx"
print_onnx_model_structure(model_path)
