import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, dim_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0) / dim_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, max_len, dim]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class SemanticEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, n_layers=3, n_heads=4, ff_dim=256, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.pos_encoder = PositionalEncoding(emb_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, src):
        # src: [batch_size, seq_len]
        x = self.embedding(src)             # [batch_size, seq_len, emb_dim]
        x = self.pos_encoder(x)             # Add positional encoding
        x = self.transformer_encoder(x)     # [batch_size, seq_len, emb_dim]
        return x
