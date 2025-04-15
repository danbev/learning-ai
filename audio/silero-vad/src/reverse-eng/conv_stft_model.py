import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ConvSTFT(nn.Module):
    def __init__(self, filter_length=256, hop_length=128):
        super().__init__()

        self.filter_length = filter_length
        self.hop_length = hop_length

        self.register_buffer('forward_basis_buffer', torch.zeros(258, 1, 256))

        # Reflection padding as in the ONNX model
        self.padding = nn.ReflectionPad1d(64)

    def transform_(self, input_data):
        padded_data = self.padding(input_data)

        padded_data = padded_data.unsqueeze(1)

        forward_transform = F.conv1d(
            padded_data,
            self.forward_basis_buffer,
            stride=self.hop_length,
            padding=0
        )

        # Split real and imaginary parts exactly as in ONNX
        cutoff = self.filter_length // 2 + 1
        real_part = forward_transform[:, :cutoff, :].float()
        imag_part = forward_transform[:, cutoff:, :].float()

        # Calculate magnitude as in ONNX
        magnitude = torch.sqrt(real_part**2 + imag_part**2)

        return magnitude

    def forward(self, input_data):
        return self.transform_(input_data)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        # Use reparam_conv name to match state dict
        self.reparam_conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=1,  # ONNX has padding=1
            stride=stride
        )
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.reparam_conv(x)
        x = self.activation(x)
        return x


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Use parameter names matching the state dict
        self.weight_ih = nn.Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.Tensor(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.Tensor(4 * hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, x, hidden):
        h_prev, c_prev = hidden
        batch_size = x.size(0)

        # Important: ONNX slices weight matrices in specific ways
        # We need to match these exactly
        i_gate_weight = self.weight_ih[:self.hidden_size]
        f_gate_weight = self.weight_ih[self.hidden_size:2*self.hidden_size]
        c_gate_weight = self.weight_ih[2*self.hidden_size:3*self.hidden_size]
        o_gate_weight = self.weight_ih[3*self.hidden_size:]

        i_gate_weight_h = self.weight_hh[:self.hidden_size]
        f_gate_weight_h = self.weight_hh[self.hidden_size:2*self.hidden_size]
        c_gate_weight_h = self.weight_hh[2*self.hidden_size:3*self.hidden_size]
        o_gate_weight_h = self.weight_hh[3*self.hidden_size:]

        i_gate_bias = self.bias_ih[:self.hidden_size]
        f_gate_bias = self.bias_ih[self.hidden_size:2*self.hidden_size]
        c_gate_bias = self.bias_ih[2*self.hidden_size:3*self.hidden_size]
        o_gate_bias = self.bias_ih[3*self.hidden_size:]

        i_gate_bias_h = self.bias_hh[:self.hidden_size]
        f_gate_bias_h = self.bias_hh[self.hidden_size:2*self.hidden_size]
        c_gate_bias_h = self.bias_hh[2*self.hidden_size:3*self.hidden_size]
        o_gate_bias_h = self.bias_hh[3*self.hidden_size:]

        # Compute gate values exactly as in the ONNX model
        # Use torch.matmul for exact operation matching
        i_gate = torch.matmul(x, i_gate_weight.t()) + i_gate_bias + \
                 torch.matmul(h_prev, i_gate_weight_h.t()) + i_gate_bias_h

        f_gate = torch.matmul(x, f_gate_weight.t()) + f_gate_bias + \
                 torch.matmul(h_prev, f_gate_weight_h.t()) + f_gate_bias_h

        c_gate = torch.matmul(x, c_gate_weight.t()) + c_gate_bias + \
                 torch.matmul(h_prev, c_gate_weight_h.t()) + c_gate_bias_h

        o_gate = torch.matmul(x, o_gate_weight.t()) + o_gate_bias + \
                 torch.matmul(h_prev, o_gate_weight_h.t()) + o_gate_bias_h

        # Apply activations with precise operations
        i_gate = torch.sigmoid(i_gate)
        f_gate = torch.sigmoid(f_gate)
        c_gate = torch.tanh(c_gate)
        o_gate = torch.sigmoid(o_gate)

        # Compute next cell and hidden state
        c_next = f_gate * c_prev + i_gate * c_gate
        h_next = o_gate * torch.tanh(c_next)

        return h_next, c_next


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.rnn = LSTMCell(input_size, hidden_size)

        # Decoder sequence as in ONNX
        self.decoder = nn.Sequential(
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Conv1d(hidden_size, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x, state):
        # Apply LSTM cell
        h, c = self.rnn(x, state)

        # Prepare for Conv1d (add time dimension)
        h_out = h.unsqueeze(2)

        # Apply decoder
        out = self.decoder(h_out)

        return out, (h, c)


class SileroVAD(nn.Module):
    def __init__(self):
        super().__init__()

        # STFT
        self.stft = ConvSTFT(filter_length=256, hop_length=128)

        # Encoder with exact parameters
        self.encoder = nn.Sequential(
            EncoderBlock(129, 128, stride=1),
            EncoderBlock(128, 64, stride=2),
            EncoderBlock(64, 64, stride=2),
            EncoderBlock(64, 128, stride=1)
        )

        # Decoder
        self.decoder = Decoder(128, 128)

        # State handling
        self._state = None
        self._context = torch.zeros(1, 0)
        self.context_size_samples = 64

    def forward(self, x, state=None, sr=16000):
        batch_size = x.shape[0]

        # Verify input
        if x.shape[-1] != 512:
            raise ValueError(f"Expected 512 samples, got {x.shape[-1]}")

        # Initialize/update context
        if len(self._context) == 0 or self._context.shape[0] != batch_size:
            self._context = torch.zeros(batch_size, self.context_size_samples, device=x.device)

        # Concatenate context with input
        x_with_context = torch.cat([self._context, x], dim=1)

        # STFT
        features = self.stft(x_with_context)

        # Encoder
        encoded = self.encoder(features)

        # Get first time step for decoder input
        decoder_input = encoded[:, :, 0]

        # Initialize state if needed
        if self._state is None or state is not None:
            if state is None:
                h = torch.zeros(batch_size, 128, device=x.device)
                c = torch.zeros(batch_size, 128, device=x.device)
                self._state = (h, c)
            else:
                self._state = state

        # Run decoder
        output, new_state = self.decoder(decoder_input, self._state)

        # Update state and context for next call
        self._state = new_state
        #self._context = x_with_context[:, -self.context_size_samples:]

        # Final output processing - mean reduction
        final_output = output.mean(dim=2).squeeze()

        return final_output, new_state
