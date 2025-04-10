import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvSTFT(nn.Module):
    def __init__(self):
        super().__init__()
        # Registration of the basis buffer
        self.register_buffer('forward_basis_buffer', torch.zeros(258, 1, 256))
        # Padding as described in your documentation
        self.padding = nn.ReflectionPad1d(64)  # Adds 64 samples of padding

    def forward(self, x):
        # Apply padding (512 samples -> 512+64*2 = 640 samples)
        x = self.padding(x)  # [B, 640]

        # Add channel dimension for convolution
        x = x.unsqueeze(1)  # [B, 1, 640]

        # Apply convolution for STFT
        x = F.conv1d(x, self.forward_basis_buffer, stride=128)  # [B, 258, 4]

        # Slice to get the first 129 channels (real components)
        x = x[:, :129, :]  # [B, 129, 4]

        return x

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.reparam_conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=1, stride=stride)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.reparam_conv(x)
        x = self.activation(x)
        return x

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.rnn = nn.LSTMCell(input_size, hidden_size)
        self.decoder = nn.Sequential(
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Conv1d(hidden_size, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x, h=None, c=None):
        batch_size = x.shape[0]

        # Initialize hidden states if needed
        if h is None:
            h = torch.zeros(batch_size, 128, device=x.device)
        if c is None:
            c = torch.zeros(batch_size, 128, device=x.device)

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

        # STFT using convolution
        self.stft = ConvSTFT()

        # Encoder with 4 convolutional layers
        self.encoder = nn.Sequential(
            EncoderBlock(129, 128, stride=2),  # Matches expected 129 input channels
            EncoderBlock(128, 64, stride=2),
            EncoderBlock(64, 64),
            EncoderBlock(64, 128)
        )

        # Decoder
        self.decoder = Decoder(128, 128)

        # For storing intermediate values
        self.debug_outputs = {}

    def forward(self, x, sr=16000):
        # Check input shape
        if sr == 16000 and x.shape[-1] != 512:
            raise ValueError(f"For 16kHz, input should have 512 samples but got {x.shape[-1]}")
        elif sr == 8000 and x.shape[-1] != 256:
            raise ValueError(f"For 8kHz, input should have 256 samples but got {x.shape[-1]}")

        # Store input for debugging
        self.debug_outputs['input'] = x

        # STFT - output shape will be [B, 129, 4]
        stft_out = self.stft(x)
        self.debug_outputs['stft_out'] = stft_out

        # Encoder processes [B, 129, 4] -> [B, C, 4]
        encoder_out = self.encoder(stft_out)
        self.debug_outputs['encoder_out'] = encoder_out

        # Get features from last time step for LSTM: [B, C, 4] -> [B, C]
        features = encoder_out[:, :, -1]
        self.debug_outputs['features'] = features

        # Decoder
        output, (h, c) = self.decoder(features)
        self.debug_outputs['lstm_h'] = h
        self.debug_outputs['lstm_c'] = c

        # Output is [B, 1, 1], reshape to [B, 1]
        return output.squeeze(2)
