# scripts/build_xi_table.py
import sys
import os

# Add parent directory (project root) to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json, torch, numpy as np
from models.full_model import SemanticCommSystem
from utilss.tokenizer import SimpleTokenizer
from utilss.metrics   import compute_semantic_similarity

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = SimpleTokenizer()
tokenizer.load("data/processed/")
vocab_size = len(tokenizer.word2idx) 
model = SemanticCommSystem(vocab_size).to(DEVICE).eval()



k_list   = range(1, 6)           # 1…5 symbols/word
snr_grid = range(-10, 21, 2)     # –10…20 dB every 2 dB
sent     = "Test sentence for similarity."  # any fixed probe

table = {}
for k in k_list:
    model.cnn_enc.k_sym = k
    for snr in snr_grid:
        # inject synthetic noise
        with torch.no_grad():
            ids = torch.tensor(tokenizer.encode(sent)).unsqueeze(0)
            logits = model(ids, ids[:, :-1], k_sym=k, snr_db=snr)
            pred_ids = torch.argmax(logits, -1)[0].tolist()
            pred = tokenizer.decode(pred_ids)   # adapt channel layer
        xi = compute_semantic_similarity(sent, pred)
        table[f"{k},{snr}"] = xi
with open("data/xi_table.json", "w") as f:
    json.dump(table, f, indent=2)
print("Table size:", len(table))
print("✅  lookup table saved → data/xi_table.json")

