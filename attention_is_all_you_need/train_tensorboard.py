# train_tensorboard.py
import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset

from tokenizer import SimpleTokenizer
from dataset import LMWindowDataset
from model import GPTModel

def sample_generate(model, tokenizer, prompt, max_new_tokens=40, temperature=1.0, top_k=None, device="cpu"):
    model.eval()
    with torch.no_grad():
        ids = tokenizer.encode(prompt)
        if len(ids) == 0:
            ids = [tokenizer.stoi.get("the", tokenizer.stoi.get(tokenizer.unk_token, 0))]
        ids = torch.tensor([ids], dtype=torch.long, device=device)
        for _ in range(max_new_tokens):
            logits, _ = model(ids)
            next_logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(next_logits, top_k)
                minv = v[:, -1].unsqueeze(1)
                next_logits = torch.where(next_logits < minv, torch.full_like(next_logits, -1e10), next_logits)
            probs = torch.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            ids = torch.cat([ids, next_id], dim=1)
        return tokenizer.decode(ids[0].tolist())

def get_lr(step, args):
    if step < args.warmup_steps:
        return args.lr * (step / max(1, args.warmup_steps))
    return args.lr * (0.1 ** (step // 10000))

def clip_gradients(model, max_norm):
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

def train(args):
    device = args.device
    os.makedirs(args.out_dir, exist_ok=True)

    # -- pick dataset --
    if args.use_small:
        print("Loading small dataset (wikitext-2 slice) for quick testing...")
        ds = load_dataset("wikitext", "wikitext-2-raw-v1")
        # join first N lines to single large text
        text = " ".join(ds["train"]["text"][: args.small_lines])
    else:
        print("Loading WikiText-103 (this can be large) ...")
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
        text = " ".join(ds["train"]["text"])

    print(f"Dataset characters: {len(text):,}")

    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab([text], vocab_size=args.vocab_size)
    tokenizer.save(os.path.join(args.out_dir, "tokenizer.json"))
    pad_index = tokenizer.stoi[tokenizer.pad_token]

    token_ids = tokenizer.encode(text)
    print(f"Total tokens: {len(token_ids):,}")

    dataset = LMWindowDataset(token_ids, block_size=args.block_size, pad_index=pad_index, stride=args.stride)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = GPTModel(
        vocab_size=len(tokenizer.stoi),
        d_model=args.d_model,
        d_ff=args.d_ff,
        d_k=args.d_k,
        d_v=args.d_v,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        pad_index=pad_index,
        device=device
    ).to(device)

    try:
        model.projection.weight = model.decoder.tgt_emb.weight
    except Exception:
        pass

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_index)
    scaler = torch.cuda.amp.GradScaler() if device.startswith("cuda") else None

    # try to load checkpoint if present
    last_ckpt = os.path.join(args.out_dir, "model_epoch_last.pt")
    start_epoch = 0
    if os.path.exists(last_ckpt):
        try:
            state = torch.load(last_ckpt, map_location=device)
            model.load_state_dict(state)
            print("Loaded checkpoint:", last_ckpt)
        except Exception as e:
            print("Failed loading checkpoint:", e)

    tb_dir = os.path.join(args.out_dir, "runs")
    writer = SummaryWriter(tb_dir)
    print("TensorBoard logs at:", tb_dir)

    global_step = 0
    model.train()
    for epoch in range(start_epoch, args.epochs):
        epoch_loss = 0.0
        t0 = time.time()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                logits, attn_maps = model(xb)
                loss = loss_fn(logits.view(-1, logits.size(-1)), yb.view(-1))

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                clip_gradients(model, args.gradient_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                clip_gradients(model, args.gradient_clip)
                optimizer.step()

            # LR scheduler (simple)
            lr = get_lr(global_step, args)
            for g in optimizer.param_groups:
                g["lr"] = lr

            epoch_loss += loss.item()
            global_step += 1

            if global_step % args.log_interval == 0:
                writer.add_scalar("train/loss_step", loss.item(), global_step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
                print(f"[{epoch}] step {global_step} loss {loss.item():.4f}")

            if global_step % args.sample_interval == 0:
                sample = sample_generate(model, tokenizer, args.sample_prompt, max_new_tokens=args.sample_tokens,
                                         temperature=args.temperature, top_k=args.top_k, device=device)
                writer.add_text("samples/generated", sample, global_step)
                writer.flush()
                print("SAMPLE:", sample)

            # early-stop quick debug
            if args.early_steps and global_step >= args.early_steps:
                print("Early stopping (debug mode).")
                break

        avg_loss = epoch_loss / max(1, len(loader))
        writer.add_scalar("train/loss_epoch", avg_loss, epoch)
        print(f"Epoch {epoch} finished in {time.time()-t0:.1f}s avg_loss {avg_loss:.4f}")

        # save checkpoint
        ckpt = os.path.join(args.out_dir, f"model_epoch{epoch}.pt")
        torch.save(model.state_dict(), ckpt)
        torch.save(model.state_dict(), last_ckpt)
        print("Saved checkpoint:", ckpt)

        if args.early_steps and global_step >= args.early_steps:
            break

    writer.close()
    final_path = os.path.join(args.out_dir, "model_final.pt")
    torch.save(model.state_dict(), final_path)
    print("Training finished. Final model saved to", final_path)
    print("TensorBoard logs at:", tb_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="out")
    parser.add_argument("--use_small", action="store_true", help="Use small wikitext-2 slice for quick testing")
    parser.add_argument("--small_lines", type=int, default=10000, help="Lines to use when --use_small is set")
    parser.add_argument("--block_size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--d_ff", type=int, default=1024)
    parser.add_argument("--d_k", type=int, default=64)
    parser.add_argument("--d_v", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--vocab_size", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--gradient_clip", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--tb_attn_interval", type=int, default=200)
    parser.add_argument("--sample_interval", type=int, default=200)
    parser.add_argument("--sample_prompt", type=str, default="The quick brown fox")
    parser.add_argument("--sample_tokens", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--visualize_layer", type=int, default=0)
    parser.add_argument("--early_steps", type=int, default=0, help="If >0 stop after this many steps (debug)")
    args = parser.parse_args()

    # default stride: half block if not specified
    if args.stride is None:
        args.stride = max(1, args.block_size // 2)

    train(args)
