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
        pe = pe.unsqueeze(0)  # shape [1, max_len, dim_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class SemanticDecoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, n_layers=3, n_heads=4, ff_dim=256, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.pos_encoder = PositionalEncoding(emb_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=emb_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.output_layer = nn.Linear(emb_dim, vocab_size)

    def forward(self, tgt_tokens, memory):
        """
        tgt_tokens: [batch, tgt_seq_len]
        memory: [batch, src_seq_len, emb_dim] — output from CNNDecoder
        """
        tgt = self.embedding(tgt_tokens)       # [batch, tgt_seq_len, emb_dim]
        tgt = self.pos_encoder(tgt)
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask)
        return self.output_layer(output)       # [batch, tgt_seq_len, vocab_size]

    def generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)
