import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
from conv_stft_model import SileroVAD
from silero_vad import read_audio, get_speech_timestamps

model = SileroVAD()
state_dict = torch.load("silero_vad_conv_pytorch.pth", map_location='cpu')
model.load_state_dict(state_dict)
model.eval()

# Create a wrapper to match the interface expected by get_speech_timestamps
class ModelWrapper:
    def __init__(self, model):
        self.model = model
        self.reset_states()

    def __call__(self, chunk, sampling_rate):
        if not torch.is_tensor(chunk):
            chunk = torch.tensor(chunk, dtype=torch.float32)

        if len(chunk.shape) == 1:
            chunk = chunk.unsqueeze(0)

        prob, self.state = self.model(chunk, self.state, sampling_rate)
        return torch.tensor([prob])

    def reset_states(self):
        # Reset LSTM state
        self.state = None

# Create the wrapper around your model
wrapped_model = ModelWrapper(model)

# Now you can use the wrapped model with get_speech_timestamps
def detect_speech(audio_file, threshold=0.5, neg_threshold=None,
                  min_speech_duration_ms=250, max_speech_duration_s=float('inf'),
                  min_silence_duration_ms=100, speech_pad_ms=30):
    """
    Detect speech segments in an audio file using your custom SileroVAD model.

    Args:
        audio_file: Path to audio file or numpy array/torch tensor of audio samples
        threshold: Speech threshold (default 0.5)
        neg_threshold: Negative threshold (default = threshold - 0.15)
        min_speech_duration_ms: Minimum speech duration in ms (default 250)
        max_speech_duration_s: Maximum speech duration in seconds (default inf)
        min_silence_duration_ms: Minimum silence duration in ms (default 100)
        speech_pad_ms: Padding for speech segments in ms (default 30)

    Returns:
        List of dictionaries with 'start' and 'end' keys for speech segments
    """
    # Load audio if it's a file path
    if isinstance(audio_file, str):
        # You'll need a function to load audio, for example:
        # audio = load_audio(audio_file)
        # For this example, let's assume we have a function that returns a tensor:
        audio = torch.tensor(np.load(audio_file) if audio_file.endswith('.npy') 
                           else np.loadtxt(audio_file))
    else:
        # Convert to tensor if it's not already
        audio = torch.tensor(audio_file) if not torch.is_tensor(audio_file) else audio_file

    # Make sure audio is 1D
    if len(audio.shape) > 1:
        audio = audio.squeeze()

    # Get speech timestamps
    speech_timestamps = get_speech_timestamps(
        audio=audio,
        model=wrapped_model,
        threshold=threshold,
        neg_threshold=neg_threshold,
        min_speech_duration_ms=min_speech_duration_ms,
        max_speech_duration_s=max_speech_duration_s,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=speech_pad_ms,
        sampling_rate=16000  # Your model is trained for 16kHz
    )

    return speech_timestamps

def main():
    print("Speech Activity Detection (VAD) Example")
    audio_data = read_audio('jfk.wav')
    print(f"Audio data shape: {audio_data.shape}, duration: {len(audio_data)/16000:.2f} seconds")
    #audio_data = torch.randn(16000 * 10)  # 10 seconds of random audio for demo

    # Detect speech
    speech_segments = detect_speech(
        audio_data,
        threshold=0.5,  # Adjust based on your model's behavior
        min_speech_duration_ms=250,
        min_silence_duration_ms=100
    )

    print("Detected speech segments:")
    for i, segment in enumerate(speech_segments):
        start_sec = segment['start'] / 16000
        end_sec = segment['end'] / 16000
        duration = end_sec - start_sec
        print(f"Segment {i+1}: {start_sec:.2f}s - {end_sec:.2f}s (duration: {duration:.2f}s)")

if __name__=="__main__":
    main()
