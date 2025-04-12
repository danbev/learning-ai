import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvSTFT(nn.Module):
    def __init__(self, filter_length=256, hop_length=128):
        super().__init__()
        self.register_buffer('forward_basis_buffer', torch.zeros(258, 1, 256))
        self.padding = nn.ReflectionPad1d(64)
        self.filter_length = filter_length
        self.hop_length = hop_length

    def transform_(self, input_data):
        # Apply padding
        input_data = self.padding(input_data)

        # Add channel dimension
        input_data = torch.unsqueeze(input_data, 1)

        # Apply convolution
        forward_transform = torch.conv1d(
            input_data,
            self.forward_basis_buffer,
            None,
            [self.hop_length],  # Stride
            [0]  # No additional padding
        )

        # Calculate cutoff for real/imaginary parts
        cutoff = int(self.filter_length // 2 + 1)

        # Extract real and imaginary parts
        real_part = forward_transform[:, :cutoff, :].float()
        imag_part = forward_transform[:, cutoff:, :].float()

        # Calculate magnitude and phase
        magnitude = torch.sqrt(real_part**2 + imag_part**2)
        phase = torch.atan2(imag_part.data, real_part.data)

        return magnitude, phase

    def forward(self, input_data):
        # Get magnitude and phase
        magnitude, _ = self.transform_(input_data)

        # Only return magnitude for the encoder
        return magnitude

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.se = nn.Identity()
        self.activation = nn.ReLU()
        self.reparam_conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=1, stride=stride)

    def forward(self, x):
        # Follow the exact order from the original code
        x = self.se(x)            # Then apply se (Identity in this case)
        x = self.reparam_conv(x)  # First apply conv
        x = self.activation(x)    # Finally apply activation
        return x

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.rnn = nn.LSTMCell(input_size, hidden_size, bias=True)
        self.decoder = nn.Sequential(
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Conv1d(hidden_size, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x, state):
        batch_size = x.shape[0]

        h, c = state

        # Process with LSTM
        h, c = self.rnn(x, (h, c))

        # Format for decoder
        h_out = h.unsqueeze(2)  # [B, H, 1]

        # Pass through decoder
        out = self.decoder(h_out)

        return out, (h, c)

class SileroVAD(nn.Module):
    def __init__(self):
        super().__init__()

        # STFT and encoder components for 16kHz processing
        self.stft = ConvSTFT(filter_length=256, hop_length=128)
        self.encoder = nn.Sequential(
            EncoderBlock(129, 128),
            EncoderBlock(128, 64),
            EncoderBlock(64, 64),
            EncoderBlock(64, 128)
        )
        self.decoder = Decoder(128, 128)

        # Context handling - key component from original model
        self._context = torch.zeros(1, 0)  # Will be resized properly on first call
        self._state = None  # LSTM state
        self._last_sr = None  # Track sample rate changes
        self._last_batch_size = None  # Track batch size changes

        # Store context size
        self.context_size_samples = 64  # Adjust this value based on original model

    def reset_states(self):
        self._context = torch.zeros(1, 0)
        self._state = None
        return None

    def forward(self, x, state=None, sr=16000):
        num_samples = 512
        if x.shape[-1] != num_samples:
            raise ValueError(f"Provided number of samples is {x.shape[-1]} (Supported values: 512 for 16000)")

        batch_size = x.shape[0]

        context_size = self.context_size_samples

        if self._last_sr and self._last_sr != sr:
            self.reset_states()

        # Reset states if batch size changed
        if self._last_batch_size and self._last_batch_size != batch_size:
            self.reset_states()

        # Initialize context if needed
        if len(self._context) == 0:
            self._context = torch.zeros(batch_size, context_size, device=x.device)

        # Ensure context is on same device as input
        #self._context = self._context.to(x.device)

        # Concatenate context and current input
        x_with_context = torch.cat([self._context, x], dim=1)

        # Process through model (this is where we need the actual processing)
        # Use STFT, encoder and decoder
        features = self.stft(x_with_context)
        encoded = self.encoder(features)

        # Select the last time step features for the LSTM
        #decoder_input = encoded[:, :, 0]  # Shape becomes [1, 128]
        decoder_input = encoded[:, :, 0]  # Shape becomes [1, 128]

        # Initialize state if not provided
        if self._state is None:
            print("Initializing state")
            h = torch.zeros(batch_size, 128, device=x.device)
            c = torch.zeros(batch_size, 128, device=x.device)
            self._state = (h, c)

        # Run decoder
        output, new_state = self.decoder(decoder_input, self._state)
        self._state = new_state

        # Update context for next call - save last context_size samples
        self._context = x_with_context[:, -context_size:]

        # Update tracking variables
        self._last_sr = sr
        self._last_batch_size = batch_size

        # Return the output and squeeze dimensions if needed
        return output.squeeze(), new_state
