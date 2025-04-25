from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sentence_transformers import SentenceTransformer, util

bleu_smoother = SmoothingFunction().method4
bert_model = SentenceTransformer("all-MiniLM-L6-v2")

def compute_bleu(reference: str, prediction: str) -> float:
    ref_tokens = reference.lower().split()
    pred_tokens = prediction.lower().split()
    return sentence_bleu([ref_tokens], pred_tokens, smoothing_function=bleu_smoother)

def compute_semantic_similarity(reference: str, prediction: str) -> float:
    ref_embed = bert_model.encode(reference, convert_to_tensor=True)
    pred_embed = bert_model.encode(prediction, convert_to_tensor=True)
    return float(util.cos_sim(ref_embed, pred_embed).item())
