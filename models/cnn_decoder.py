
import torch
import torch.nn as nn

class CNNDecoder(nn.Module):
    def __init__(self, in_channels=16, hidden_channels=64, out_channels=128, kernel_size=3):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=kernel_size, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, out_channels, kernel_size=kernel_size, padding=1),
            nn.ReLU()
        )

    def forward(self,x): 
   
        x = x.permute(0, 2, 1)     # → [batch_size, in_channels, seq_len]
        x = self.decoder(x)        # → [batch_size, out_channels, seq_len]
        x = x.permute(0, 2, 1)
      
        return x
