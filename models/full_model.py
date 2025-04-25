# models/full_model.py
import torch.nn as nn
from models.semantic_encoder import SemanticEncoder
from models.cnn_encoder import CNNEncoder
from models.channel_layer import ChannelLayer
from models.cnn_decoder import CNNDecoder
from models.semantic_decoder import SemanticDecoder
import torch.nn.functional as F
class SemanticCommSystem(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.encoder = SemanticEncoder(vocab_size)
        self.cnn_enc = CNNEncoder()
        self.channel = ChannelLayer(snr_db=10, rician_k=3.0)
        self.k2fixed   = nn.Conv1d(   # 1×1 conv
            in_channels=1, out_channels=16, kernel_size=1)
        self.cnn_dec = CNNDecoder()
        self.decoder = SemanticDecoder(vocab_size)

    def forward(self, src, tgt = None, *, k_sym=None, snr_db=10.0):
        """
        src : [B, L]  token ids
        tgt : [B, L-1] teacher-forced tokens
        k_sym : optional int   symbols/word for this pass
        snr_db: float          per-batch SNR fed into ChannelLayer
        """
        if tgt is None:                       # fallback: use src shifted
            tgt = src[:, :-1] if src.size(1) > 1 else src
        x = self.encoder(src)                         # transformer encoder

        # pass k_sym down to CNN encoder
        x = self.cnn_enc(x, k_sym=k_sym)

        # ------------- maritime channel -----------------
        x = self.channel(x, snr_db=snr_db)            # ⭐ now uses arg
        # ------------------------------------------------
        if k_sym is not None and x.size(-1) < 16:
            pad_c = 16 - x.size(-1)
            x = F.pad(x, (0, pad_c))
        x = self.cnn_dec(x)
        logits = self.decoder(tgt, x)
        return logits