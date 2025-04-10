import torch
import numpy as np
import matplotlib.pyplot as plt
from conv_stft_model import SileroVAD

def test_conv_stft():
    # Create model
    model = SileroVAD()
    model.eval()

    # Create sample input
    input_tensor = torch.randn(1, 512, dtype=torch.float32)
    print(f"Input tensor shape: {input_tensor.shape}")

    # Test forward pass
    with torch.no_grad():
        # Run the model without loading weights - just to test shapes
        output = model(input_tensor)

        # Print shapes of all intermediate tensors
        print("\nModel forward pass shapes:")
        for name, tensor in model.debug_outputs.items():
            print(f"  {name}: {tensor.shape}")

    # Visualize the STFT output if available
    if 'stft_out' in model.debug_outputs:
        plt.figure(figsize=(10, 6))
        stft_data = model.debug_outputs['stft_out'][0].numpy()
        plt.imshow(stft_data, aspect='auto', origin='lower')
        plt.colorbar()
        plt.title('ConvSTFT Output')
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        plt.savefig('conv_stft_output.png')
        print("Saved STFT visualization to conv_stft_output.png")

if __name__ == "__main__":
    test_conv_stft()
