from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
import onnx

#model_path = "/home/danbev/work/ai/audio/silero-vad/src/silero_vad/data/silero_vad.onnx" 
#model = onnx.load(model_path)
#print(model)

model = load_silero_vad(onnx=False)
#print(model)
#model.code()

# Print the model's graph
#print(model.graph)
# Print the model's graph
print(model.code)
state_dict = model.state_dict()

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

members = [attr for attr in dir(model) if not attr.startswith('__')]
print(members)
#print(model.named_parameters)
print(model.modules)

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


wav = read_audio('jfk.wav')
#wav = read_audio('silence.wav')

#speech_timestamps = get_speech_timestamps(
#  wav,
#  model,
#  return_seconds=True,
#)
#print(speech_timestamps)

predicts = model.audio_forward(wav, sr=16000)

for i, value in enumerate(predicts[0]):
    print(f"[{i}] probability: {value:.10f}")

#print(model.state_dict)
