import torch
import soundfile as sf
from transformers import AutoModelForAudioFrameClassification, AutoProcessor
import numpy as np

np.set_printoptions(suppress=True, precision=8, linewidth=200)


model_id = "nvidia/Nemotron-3-Diarization"

print("Loading processor and model...")
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForAudioFrameClassification.from_pretrained(
    model_id,
    device_map="auto",
    dtype=torch.float32  # or torch.float16 if running on GPU
)

# Load 16 kHz mono audio
audio_path = "samples/conversation.wav"
audio, sr = sf.read(audio_path)
if sr != 16000:
    import librosa
    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    sr = 16000

# Preprocess to Mel features
inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
inputs = inputs.to(model.device, dtype=model.dtype)

print(f"Input features shape: {inputs.input_features.shape}")

with torch.inference_mode():
    outputs = model(**inputs)
    logits = outputs.logits  # Shape: (1, num_frames, 8) — 1 frame every 10 ms

# Inspect raw frame probabilities
probs = torch.sigmoid(logits)
print(f"Output logits shape: {logits.shape}")  # [1, T, 8]

probs_np = probs[0].cpu().numpy()

start_frame = 25
end_frame = 45

print(f"{'Frame':>6} | {'Time':>7} | Speaker Probabilities (Channels 0 to 7)")
print("-" * 105)

for frame_idx in range(start_frame, min(end_frame, probs_np.shape[0])):
    timestamp = frame_idx * 0.010  # 10 ms per frame
    probs_str = "  ".join(f"{val:10.8f}" for val in probs_np[frame_idx])
    print(f"{frame_idx:6d} | {timestamp:6.2f}s | {probs_str}")

# Convert logits to segment intervals
segments = processor.extract_speaker_dict(logits, inputs.attention_mask)[0]
for seg in segments:
    print(f"Speaker {seg['Speaker']}: {seg['Start']:.2f}s - {seg['End']:.2f}s")

