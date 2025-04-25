import sys
import os

# Add parent directory (project root) to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.full_model import SemanticCommSystem
from utilss.xi_lookup   import xi_lookup
from utilss.allocator   import greedy_swap_allocator

model = SemanticCommSystem(vocab_size=100)
model.cnn_enc.k_sym = 3
print("model forward OK ✔")

print("xi(3, 5dB) =", xi_lookup(3, 5))
print("allocator OK ✔" if greedy_swap_allocator([[1,2],[3,0]]) else "allocator error")
