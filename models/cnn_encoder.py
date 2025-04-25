import torch
import torch.nn as nn

class CNNEncoder(nn.Module):
    def __init__(self, in_channels=128, hidden_channels=64, out_channels=16,
                 kernel_size=3, k_sym=3):          # new arg
        super().__init__()
        self.k_sym = k_sym

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=kernel_size, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, out_channels, kernel_size=kernel_size, padding=1),
            nn.ReLU()
        )
        

    def forward(self, x, k_sym=None):
        k = k_sym if k_sym is not None else self.k_sym
        # x: [batch_size, seq_len, in_channels]
        x = x.permute(0, 2, 1)  # → [batch_size, in_channels, seq_len]
        x = self.encoder(x)     # → [batch_size, out_channels, seq_len]
        x = x.permute(0, 2, 1)  # → [batch_size, seq_len, out_channels]
        out = x[:, :k, :]
        return out
        
