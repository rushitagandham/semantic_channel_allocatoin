import sys
import os

# Add parent directory (project root) to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.nn.utils.rnn import pad_sequence 
from utilss.tokenizer import SimpleTokenizer
from models.full_model import SemanticCommSystem
from utilss.metrics import compute_bleu, compute_semantic_similarity
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------ Load Assets ------------ #
tokenizer = SimpleTokenizer()
tokenizer.load("data/processed/")

model = SemanticCommSystem(vocab_size=len(tokenizer.word2idx)).to(DEVICE)
model.load_state_dict(torch.load("model_checkpoint.pt"))  # Replace with your saved model
model.eval()

# ------------ Load Test Sentences ------------ #
with open("data/processed/maritime_sentences.txt", "r", encoding="utf-8") as f:
    sentences = f.read().splitlines()

test_sentences = sentences[:100]  # Evaluate on a sample

# ------------ Evaluation ------------ #
bleu_scores = []
semantic_scores = []

for sentence in tqdm(test_sentences):
    input_ids = tokenizer.encode(sentence)[:30]
    src = torch.tensor(input_ids).unsqueeze(0).to(DEVICE)       # [1, seq_len]
    tgt = torch.tensor(input_ids[:-1]).unsqueeze(0).to(DEVICE)  # [1, seq_len-1]

    with torch.no_grad():
        logits = model(src, tgt, k_sym = 16)
        pred_ids = torch.argmax(logits, dim=-1)[0]  # [seq_len]
        pred_sentence = tokenizer.decode(pred_ids.tolist())

    bleu = compute_bleu(sentence, pred_sentence)
    sem = compute_semantic_similarity(sentence, pred_sentence)

    bleu_scores.append(bleu)
    semantic_scores.append(sem)

    # Print a few examples
    if len(bleu_scores) < 5:
        print("\n--- SAMPLE ---")
        print("Ref:", sentence)
        print("Pred:", pred_sentence)
        print("BLEU:", round(bleu, 3), "| Sim:", round(sem, 3))

# ------------ Results ------------ #
avg_bleu = sum(bleu_scores) / len(bleu_scores)
avg_sim = sum(semantic_scores) / len(semantic_scores)

print("\n===== FINAL RESULTS =====")
print(f"Average BLEU: {avg_bleu:.4f}")
print(f"Average Semantic Similarity: {avg_sim:.4f}")
