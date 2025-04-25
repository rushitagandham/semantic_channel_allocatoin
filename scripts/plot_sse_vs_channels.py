import sys
import os

# Add parent directory (project root) to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np, matplotlib.pyplot as plt, json, torch
from utilss.metrics import compute_semantic_similarity
from utilss.allocator import greedy_swap_allocator
from utilss.xi_lookup import xi_lookup
from models.full_model import SemanticCommSystem
from utilss.tokenizer import SimpleTokenizer

def simulate(M, N=5, k_max=5, snr_dB=5):
    k_vec = torch.randint(1, k_max+1, (N,))
    snr_mat = torch.ones(N, M)*snr_dB           # flat SNR grid
    phi = np.zeros((N, M))
    for u in range(N):
        for m in range(M):
            k_u   = int(k_vec[u].item())           # tensor → int
            snr_u = float(snr_mat[u, m].item())    # tensor → float
            phi[u, m] = xi_lookup(k_u, snr_u) / k_u

    mapping = greedy_swap_allocator(phi)
    return sum(phi[u, mapping[u]] for u in range(N) if mapping[u]!=-1)

Ms = list(range(1, 11))
sse = [simulate(M) for M in Ms]

plt.plot(Ms, sse, marker='o', label='Greedy-swap')
plt.xlabel('Number of channels M')
plt.ylabel('Sum S-SE (suts/s/Hz)')
plt.grid(True); plt.legend()
plt.tight_layout()
plt.savefig('sse_vs_channels.png')
print('Saved ➜ sse_vs_channels.png')
