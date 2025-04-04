from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

model = load_silero_vad()
print(model)
wav = read_audio('jfk.wav')

speech_timestamps = get_speech_timestamps(
  wav,
  model,
  return_seconds=True,
)

print(speech_timestamps)
