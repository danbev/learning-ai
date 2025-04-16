from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
import onnx

model = load_silero_vad(onnx=False)
state_dict = model.state_dict()

wav = read_audio('jfk.wav')
predicts = model.audio_forward(wav, sr=16000)

for i, value in enumerate(predicts[0]):
    print(f"[{i}] probability: {value:.10f}")

