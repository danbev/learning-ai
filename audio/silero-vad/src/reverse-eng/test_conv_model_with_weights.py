import torch
import os
import numpy as np
from conv_stft_model import SileroVAD
from silero_vad import read_audio

def test_model_with_weights(pytorch_model_path):
    """Test loaded model with weights"""
    # Load PyTorch model
    pytorch_model = SileroVAD()
    pytorch_model.eval()

    state_dict = torch.load(pytorch_model_path)
    #for key in state_dict.keys():
    #    if 'decoder' in key:
    #        print(key)

    if pytorch_model_path and os.path.exists(pytorch_model_path):
        print(f"Loading PyTorch model from {pytorch_model_path}")
        pytorch_model.load_state_dict(torch.load(pytorch_model_path))
    else:
        print(f"Warning: Model file not found at {pytorch_model_path}")

    #print(pytorch_model)
    #print(pytorch_model.state_dict())
    #print("STFT basis buffer samples:", pytorch_model.stft.forward_basis_buffer)

    # Create test input
    sample_rate = 16000

    # For ramdon input (just one sample frame)
    #input_tensor = torch.randn(1, 512, dtype=torch.float32)

    # Process entire .wav file in 512-sample chunks
    wav = read_audio('jfk.wav')
    print(f"Original wav shape: {wav.shape}, duration: {len(wav)/16000:.2f} seconds")

    # Calculate how many 512-sample chunks we need
    chunk_size = 512
    n_chunks = len(wav) // chunk_size
    if len(wav) % chunk_size != 0:
        n_chunks += 1  # Add one more chunk for remaining samples

    print(f"Processing {n_chunks} chunks of {chunk_size} samples each")

    # Initialize variables to track results
    all_probabilities = []

    state = None

    pytorch_model.eval()
    # Process each chunk
    with torch.no_grad():
        for i in range(n_chunks):
            # Get current chunk
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(wav))

            # Handle partial chunks at the end
            current_chunk = wav[start_idx:end_idx]
            if len(current_chunk) < chunk_size:
                # Pad with zeros if needed
                padded_chunk = torch.zeros(chunk_size)
                padded_chunk[:len(current_chunk)] = current_chunk
                current_chunk = padded_chunk

            # Add batch dimension
            input_tensor = current_chunk.unsqueeze(0)

            # Run the model
            pytorch_output, new_state = pytorch_model(input_tensor, state)

            state = new_state

            # Store the probability
            all_probabilities.append(pytorch_output.item())

            print(f"Processed chunk {i+1}/{n_chunks}, probability: {pytorch_output.item():.6f}")

    # Print summary statistics
    print("\nProcessing complete!")
    print(f"Processed {n_chunks} chunks of audio")
    print(f"Average probability: {sum(all_probabilities)/len(all_probabilities):.6f}")
    print(f"Max probability: {max(all_probabilities):.6f}")
    print(f"Min probability: {min(all_probabilities):.6f}")

if __name__ == "__main__":
    pytorch_model_path = "silero_vad_conv_pytorch.pth"

    test_model_with_weights(pytorch_model_path)
