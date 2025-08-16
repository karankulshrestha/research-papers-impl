# tokenizer.py
import json
import re
from collections import Counter

class SimpleTokenizer:
    def __init__(self, unk_token="<unk>", pad_token="<pad>"):
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.stoi = {}
        self.itos = {}

    @staticmethod
    def _pretokenize(text: str):
        # Better preprocessing for WikiText data
        # Remove excessive whitespace and normalize
        text = re.sub(r'\s+', ' ', text)
        # Split on word boundaries and punctuation, preserve important tokens
        tokens = re.findall(r"\b\w+\b|[.!?,:;'\"\-\(\)\[\]{}]", text, flags=re.UNICODE)
        return [token.lower() for token in tokens if token.strip()]

    def build_vocab(self, texts: list, vocab_size=4000, min_freq=1):
        print("Counting token frequencies...")
        counter = Counter()
        for i, t in enumerate(texts):
            if i % 1000 == 0 and i > 0:
                print(f"Processed {i} texts...")
            counter.update(self._pretokenize(t))
        
        print(f"Found {len(counter)} unique tokens before filtering")
        most_common = [tok for tok, cnt in counter.most_common(vocab_size) if cnt >= min_freq]
        
        # Add special tokens at the beginning
        vocab = [self.pad_token, self.unk_token] + most_common
        
        # Ensure we don't exceed vocab_size
        if len(vocab) > vocab_size:
            vocab = vocab[:vocab_size]
            
        self.stoi = {tok: i for i, tok in enumerate(vocab)}
        self.itos = {i: tok for tok, i in self.stoi.items()}
        print(f"Final vocabulary size: {len(self.stoi)}")

    def encode(self, text: str):
        toks = self._pretokenize(text)
        return [self.stoi.get(t, self.stoi[self.unk_token]) for t in toks]

    def decode(self, ids: list):
        toks = [self.itos.get(i, self.unk_token) for i in ids]
        out_tokens = []
        prev_was_word = False
        for tok in toks:
            is_word = bool(re.match(r"^\w+$", tok))
            if is_word:
                if prev_was_word:
                    out_tokens.append(" " + tok)
                else:
                    out_tokens.append(tok)
            else:
                out_tokens.append(tok)
            prev_was_word = is_word
        return "".join(out_tokens).strip()

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.stoi, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            stoi = json.load(f)
        tok = cls()
        tok.stoi = stoi
        tok.itos = {i: s for s, i in stoi.items()}
        return tok
