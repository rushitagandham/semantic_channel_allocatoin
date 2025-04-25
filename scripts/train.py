import os
import math
import numpy as np
from typing import List

import sys
import os

# Add parent directory (project root) to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from tqdm import tqdm

# ----------------------------
#  Project‑local imports
# ----------------------------
from models.full_model import SemanticCommSystem
from utilss.tokenizer import SimpleTokenizer
from utilss.metrics import compute_semantic_similarity
from utilss.allocator import greedy_swap_allocator
from utilss.xi_lookup import xi_lookup  # pre‑computed (k, SNR) → ξ table

# ----------------------------
#  Hyper‑parameters
# ----------------------------
BATCH_SIZE_USERS = 5      # users per mini‑batch
EPOCHS            = 10
MAX_SEQ_LEN       = 30
EMB_DIM           = 128   # must match model
K_MAX             = 5     # max semantic symbols/word
M_CHANNELS        = 5     # orthogonal channels per cell
LAMBDA_SIM        = 0.2   # weight on semantic‑sim loss
LEARNING_RATE     = 3e-4
SAVE_PATH         = "model_checkpoint.pt"
DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
#  Dataset stub (replace with your own)
# ----------------------------
class MaritimeDataset(Dataset):
    """Takes a list[str] of sentences and a tokenizer."""
    def __init__(self, sentences: List[str], tokenizer: SimpleTokenizer):
        self.sents = sentences
        self.tok   = tokenizer

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, idx):
        s = self.sents[idx]
        ids = self.tok.encode(s)[:MAX_SEQ_LEN]
        return torch.tensor(ids, dtype=torch.long)

def collate_fn(batch):
    pad = 0
    out = pad_sequence(batch, batch_first=True, padding_value=pad)
    return out.to(DEVICE)

# ----------------------------
#  Data loading
# ----------------------------
print("📖 Loading sentences …")
with open("data/processed/maritime_sentences.txt", "r", encoding="utf-8") as f:
    all_lines = f.read().splitlines()
    all_lines = all_lines[:5000]

# OPTIONAL: subsample for quick debugging
# all_lines = all_lines[:2000]

print("🔤 Loading tokenizer …")
Tok = SimpleTokenizer()
Tok.load("data/processed/")
vocab_size = len(Tok.word2idx) 
loader = DataLoader(
    MaritimeDataset(all_lines, Tok),
    batch_size=BATCH_SIZE_USERS,
    shuffle=True,
    collate_fn=collate_fn,
)

# ----------------------------
#  Model + Optimiser
# ----------------------------
print("🧠 Building model …")
model = SemanticCommSystem(vocab_size).to(DEVICE)
opt   = AdamW(model.parameters(), lr=LEARNING_RATE)
ce_loss_fn = nn.CrossEntropyLoss(ignore_index=0)
model.load_state_dict(torch.load(SAVE_PATH))

# ----------------------------
#  Helper: forward pass for one user sentence
# ----------------------------

def forward_single(sent_ids: torch.Tensor, k_sym: int, snr_db: float):
    """sent_ids: [seq] tensor of token ids (includes <SOS>, <EOS>)"""
    # 1) Set k for this user on encoder/decoder
    model.cnn_enc.k_sym = k_sym

    # 2) Build tgt (teacher‑forced) and src
    src = sent_ids.unsqueeze(0)           # [1, seq]
    tgt_in  = sent_ids[:-1].unsqueeze(0)  # remove last token
    tgt_out = sent_ids[1:].unsqueeze(0)   # remove first token
    # 3) Forward   (snr injected inside ChannelLayer via global variable)
    logits = model(src, tgt_in,k_sym=k_sym, snr_db=snr_db)  # implement snr_override in ChannelLayer
    loss_ce = ce_loss_fn(logits.view(-1, logits.size(-1)), tgt_out.view(-1))
    # 4) Semantic similarity
    pred_ids = torch.argmax(logits, dim=-1)[0].tolist()
    ref_sent = Tok.decode(sent_ids.tolist())
    pred_sent= Tok.decode(pred_ids)
    xi = compute_semantic_similarity(ref_sent, pred_sent)
    loss = loss_ce + LAMBDA_SIM * (1 - xi)
    return loss, xi

# ----------------------------
#  Training loop
# ----------------------------
print("🚀 Starting training …")
for epoch in range(1, EPOCHS+1):
    model.train()
    running_loss = 0.0
    pbar = tqdm(enumerate(loader, 1), total=len(loader), desc=f"Epoch {epoch}")

    for step, batch_ids in pbar:
        N = batch_ids.size(0)            # users in this mini‑batch
        # 1) Sample k_sym for each user
        k_vec = torch.randint(1, K_MAX+1, (N,))
        # 2) Sample SNR matrix for these users & M channels
        snr_mat = torch.randn(N, M_CHANNELS) * 5 + 5   # mean≈5 dB, std≈5
        # 3) Build Φ matrix
        phi = np.zeros((N, M_CHANNELS))
        for u in range(N):
            for m in range(M_CHANNELS):
                xi = xi_lookup(int(k_vec[u]), float(snr_mat[u, m]))
                phi[u, m] = xi / int(k_vec[u])
        # 4) Allocate channels
        mapping = greedy_swap_allocator(phi)    # list len N

        # ---------------- training users one‑by‑one -------------
        opt.zero_grad()
        batch_loss = 0.0
        for u in range(N):
            ch = mapping[u]
            if ch == -1:
                continue  # user skipped this round
            snr = float(snr_mat[u, ch])
            loss_u, _ = forward_single(batch_ids[u], int(k_vec[u]), snr)
            batch_loss += loss_u
        if batch_loss == 0:
            continue  # all skipped (unlikely but safe)
        batch_loss.backward()
        opt.step()
        running_loss += batch_loss.item()
        pbar.set_postfix(loss=f"{running_loss/step:.4f}")

    print(f"Epoch {epoch} — avg loss: {running_loss/len(loader):.4f}")
    torch.save(model.state_dict(), SAVE_PATH)
    print("💾 checkpoint saved →", SAVE_PATH)

print("✅ Training done!")
