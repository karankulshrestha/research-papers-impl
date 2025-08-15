# generate.py
import os
import torch
from tokenizer import SimpleTokenizer
from model import GPTModel

OUT_DIR = "out"
CKPT_PATH = os.path.join(OUT_DIR, "model_final.pt")
TOK_PATH = os.path.join(OUT_DIR, "tokenizer.json")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_tokenizer(path):
    return SimpleTokenizer.load(path)

def top_k_logits(logits, k):
    if k is None or k <= 0:
        return logits
    v, _ = torch.topk(logits, k)
    minv = v[:, -1].unsqueeze(1)
    return torch.where(logits < minv, torch.full_like(logits, -1e10), logits)

def generate(model, tokenizer, prompt, max_new_tokens=40, temperature=1.0, top_k=40):
    model.eval()
    with torch.no_grad():
        ids = tokenizer.encode(prompt)
        if len(ids) == 0:
            ids = [tokenizer.stoi.get("the", 0)]
        ids = torch.tensor([ids], dtype=torch.long, device=DEVICE)
        for _ in range(max_new_tokens):
            logits, _ = model(ids)
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = top_k_logits(logits, top_k)
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            ids = torch.cat([ids, next_id], dim=1)
        return tokenizer.decode(ids[0].tolist())

def main():
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"Checkpoint not found at {CKPT_PATH}")
    if not os.path.exists(TOK_PATH):
        raise FileNotFoundError(f"Tokenizer not found at {TOK_PATH}")

    tokenizer = load_tokenizer(TOK_PATH)
    vocab_size = len(tokenizer.stoi)

    # match the train-time hyperparams you used
    model = GPTModel(
        vocab_size=vocab_size,
        d_model=256,
        d_ff=1024,
        d_k=64,
        d_v=64,
        n_heads=4,
        n_layers=2,
        pad_index=tokenizer.stoi[tokenizer.pad_token],
        device=DEVICE)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
    model.to(DEVICE)

    prompt = "women"
    out = generate(model, tokenizer, prompt, max_new_tokens=80, temperature=1.0, top_k=40)
    print("=== SAMPLE ===")
    print(out)
    print("==============")

if __name__ == "__main__":
    main()
