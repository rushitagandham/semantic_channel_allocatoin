from utils.metrics import compute_bleu, compute_semantic_similarity

ref = "Tanker Blue Whale is heading to Port of LA at 12 knots."
pred = "Blue Whale vessel heading to Port of LA at speed 12 knots."

print("BLEU:", compute_bleu(ref, pred))
print("Semantic Sim:", compute_semantic_similarity(ref, pred))
