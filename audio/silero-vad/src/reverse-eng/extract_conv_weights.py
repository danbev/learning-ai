import torch
import os
from conv_stft_model import SileroVAD

def extract_weights(jit_model_path, save_pytorch_model=True):
    """Extract weights from JIT model and load them into PyTorch model"""
    print(f"Loading JIT model from {jit_model_path}")
    jit_model = torch.jit.load(jit_model_path)

    # Create our PyTorch model
    pytorch_model = SileroVAD()

    # Get state dict from JIT model
    state_dict = jit_model.state_dict()

    print(f"Found {len(state_dict)} parameters in JIT model")

    # Create mapping between JIT and PyTorch model keys
    mapping = {
        # STFT mapping
        "_model.stft.forward_basis_buffer": "stft.forward_basis_buffer",

        # Encoder mappings
        "_model.encoder.0.reparam_conv.weight": "encoder.0.reparam_conv.weight",
        "_model.encoder.0.reparam_conv.bias": "encoder.0.reparam_conv.bias",
        "_model.encoder.1.reparam_conv.weight": "encoder.1.reparam_conv.weight",
        "_model.encoder.1.reparam_conv.bias": "encoder.1.reparam_conv.bias",
        "_model.encoder.2.reparam_conv.weight": "encoder.2.reparam_conv.weight",
        "_model.encoder.2.reparam_conv.bias": "encoder.2.reparam_conv.bias",
        "_model.encoder.3.reparam_conv.weight": "encoder.3.reparam_conv.weight",
        "_model.encoder.3.reparam_conv.bias": "encoder.3.reparam_conv.bias",

        # LSTM mappings
        "_model.decoder.rnn.weight_ih": "decoder.rnn.weight_ih",
        "_model.decoder.rnn.weight_hh": "decoder.rnn.weight_hh",
        "_model.decoder.rnn.bias_ih": "decoder.rnn.bias_ih",
        "_model.decoder.rnn.bias_hh": "decoder.rnn.bias_hh",

        # Final conv mappings
        "_model.decoder.decoder.2.weight": "decoder.decoder.2.weight",
        "_model.decoder.decoder.2.bias": "decoder.decoder.2.bias",
    }

    # Create new state dict for PyTorch model
    new_state_dict = {}

    # Transfer weights
    for jit_key, pt_key in mapping.items():
        if jit_key in state_dict:
            new_state_dict[pt_key] = state_dict[jit_key]
            print(f"Transferred: {jit_key} -> {pt_key}, Shape: {state_dict[jit_key].shape}")
        else:
            print(f"Warning: {jit_key} not found in JIT model")

    # Load weights into PyTorch model
    pytorch_model.load_state_dict(new_state_dict, strict=False)

    # Check for missing keys
    expected_keys = set(dict(pytorch_model.named_parameters()).keys())
    loaded_keys = set(new_state_dict.keys())
    missing_keys = expected_keys - loaded_keys

    if missing_keys:
        print(f"Warning: {len(missing_keys)} parameters were not loaded:")
        for key in missing_keys:
            print(f"  - {key}")

    # Save the model if requested
    if save_pytorch_model:
        save_path = "silero_vad_conv_pytorch.pth"
        torch.save(pytorch_model.state_dict(), save_path)
        print(f"Saved PyTorch model to {save_path}")

    return pytorch_model

if __name__ == "__main__":
    # Check if JIT model exists
    jit_model_path = "/home/danbev/work/ai/audio/silero-vad/src/silero_vad/data/silero_vad.jit"
    if not os.path.exists(jit_model_path):
        print(f"Error: JIT model not found at {jit_model_path}")
        exit(1)

    # Extract weights
    model = extract_weights(jit_model_path)
    print("Weight extraction complete!")
