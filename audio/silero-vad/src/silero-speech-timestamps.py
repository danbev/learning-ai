from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
import onnx

model = load_silero_vad(onnx=False)
state_dict = model.state_dict()

wav = read_audio('jfk.wav')
speech_timestamps = get_speech_timestamps(
  wav,
  model,
  return_seconds=True,
)
for i, ts in enumerate(speech_timestamps):
    start = ts['start']
    end = ts['end']
    print(f"Speech segment {i}: start={start:.2f}s, end={end:.2f}s")
