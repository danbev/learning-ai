import torch
import os
import numpy as np
from conv_stft_model import SileroVAD
from silero_vad import read_audio

def test_model_with_weights(pytorch_model_path, jit_model_path=None):
    """Test loaded model with weights"""
    # Load PyTorch model
    pytorch_model = SileroVAD()
    pytorch_model.eval()

    if pytorch_model_path and os.path.exists(pytorch_model_path):
        print(f"Loading PyTorch model from {pytorch_model_path}")
        pytorch_model.load_state_dict(torch.load(pytorch_model_path))
        print("STFT basis buffer samples:", pytorch_model.stft.forward_basis_buffer[0:5, 0, 0:5])
    else:
        print(f"Warning: Model file not found at {pytorch_model_path}")


    # Load JIT model if available
    jit_model = None
    if jit_model_path and os.path.exists(jit_model_path):
        print(f"Loading JIT model from {jit_model_path}")
        jit_model = torch.jit.load(jit_model_path)

    # Create test input
    sample_rate = 16000

    # For ramdon input (just one sample frame)
    #input_tensor = torch.randn(1, 512, dtype=torch.float32)

    # One audio sample frame
    wav = read_audio('jfk.wav')
    # Extract just the first 512 samples (for 16kHz sample rate)
    # Make sure to handle tensor dimensions properly
    if wav.dim() == 1:
        # If wav is a 1D tensor, add batch dimension
        input_tensor = wav[:512].unsqueeze(0)
    elif wav.dim() == 2:
        # If wav is already 2D (batch, samples), just take first 512 samples
        input_tensor = wav[:, :512]

    # Ensure we have exactly 512 samples
    assert input_tensor.shape[-1] == 512, f"Expected 512 samples, but got {input_tensor.shape[-1]}"
    print(f"Input tensor shape: {input_tensor.shape}")

    # Run inference with PyTorch model
    with torch.no_grad():
        # Run the model
        pytorch_output = pytorch_model(input_tensor, sample_rate)

        # Print shapes of all intermediate tensors
        print("\nIntermediate shapes from PyTorch model:")
        for name, tensor in pytorch_model.debug_outputs.items():
            print(f"  {name}: {tensor.shape}")
            # Print sample values
            flat_tensor = tensor.flatten()
            if len(flat_tensor) > 0:
                print(f"    Sample values: {flat_tensor[:5]}")

        # Run JIT model if available
        if jit_model:
            jit_output = jit_model(input_tensor, sample_rate)
            print(f"\nJIT output: {jit_output.item():.6f}")
            print(f"PyTorch output: {pytorch_output.item():.6f}")
            print(f"Difference: {abs(jit_output.item() - pytorch_output.item()):.6f}")
        else:
            print(f"shape of PyTorch output: {pytorch_output.shape}")
            print(f"\nPyTorch output: {pytorch_output.item():.6f}")

if __name__ == "__main__":
    pytorch_model_path = "silero_vad_conv_pytorch.pth"
    jit_model_path = "silero_vad.jit"

    test_model_with_weights(pytorch_model_path, jit_model_path)
