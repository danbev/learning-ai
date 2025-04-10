import onnx

model_path = "/home/danbev/work/ai/audio/silero-vad/src/silero_vad/data/silero_vad.onnx" 
model = onnx.load(model_path)
print(model)
