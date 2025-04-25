# utils/build_vocab.py
from tokenizer import SimpleTokenizer

file_path = "data/processed/maritime_sentences.txt"

with open(file_path, "r", encoding="utf-8") as f:
    lines = f.read().splitlines()

tokenizer = SimpleTokenizer(min_freq=2)
tokenizer.build_vocab(lines)
tokenizer.save("data/processed/")
