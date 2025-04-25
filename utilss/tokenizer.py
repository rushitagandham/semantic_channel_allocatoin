import os
import json
from collections import Counter

SPECIAL_TOKENS = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']

class SimpleTokenizer:
    def __init__(self, min_freq=2):
        self.min_freq = min_freq
        self.word2idx = {}
        self.idx2word = {}

    def build_vocab(self, sentences):
        counter = Counter()
        for sentence in sentences:
            tokens = sentence.lower().strip().split()
            counter.update(tokens)

        vocab = [word for word, freq in counter.items() if freq >= self.min_freq]
        vocab = SPECIAL_TOKENS + sorted(vocab)

        self.word2idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        print(f"✅ Vocabulary built with {len(self.word2idx)} tokens")

    def encode(self, sentence):
        tokens = sentence.lower().strip().split()
        return [self.word2idx.get('<SOS>')] + \
               [self.word2idx.get(t, self.word2idx['<UNK>']) for t in tokens] + \
               [self.word2idx.get('<EOS>')]

    def decode(self, indices):
        words = [self.idx2word.get(idx, '<UNK>') for idx in indices]
        return ' '.join([w for w in words if w not in ['<SOS>', '<EOS>', '<PAD>']])

    def save(self, path):
        with open(os.path.join(path, "word2idx.json"), "w") as f:
            json.dump(self.word2idx, f)
        with open(os.path.join(path, "idx2word.json"), "w") as f:
            json.dump(self.idx2word, f)

    def load(self, path):
        with open(os.path.join(path, "word2idx.json"), "r") as f:
            self.word2idx = json.load(f)
        with open(os.path.join(path, "idx2word.json"), "r") as f:
            self.idx2word = {int(k): v for k, v in json.load(f).items()}

