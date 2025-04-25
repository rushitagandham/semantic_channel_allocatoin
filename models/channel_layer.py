import torch
import torch.nn as nn

class ChannelLayer(nn.Module):
    def __init__(self, snr_db=10, rician_k=5.0):
        super().__init__()
        self.snr_db = snr_db
        self.k = rician_k

    def forward(self, x, snr_db=None):
        # x: [batch_size, seq_len, channels]
        snr_db = snr_db or self.snr_db
        device = x.device
        dtype = x.dtype

        # Signal power and noise power calculation
        snr_linear = 10 ** (self.snr_db / 10)
        signal_power = torch.mean(x ** 2)
        noise_power = signal_power / snr_linear

        # Convert float scalars to tensors (on the same device)
        k = torch.tensor(self.k, device=device, dtype=dtype)
        one = torch.tensor(1.0, device=device, dtype=dtype)

        # Rician fading
        los = torch.ones_like(x)
        nlos = torch.randn_like(x)
        H = torch.sqrt(k / (k + one)) * los + torch.sqrt(one / (k + one)) * nlos

        # Apply fading
        faded = x * H

        # Add AWGN
        noise = torch.randn_like(faded) * torch.sqrt(noise_power)
        return faded + noise
