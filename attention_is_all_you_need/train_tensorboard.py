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
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import seaborn as sns

from tokenizer import SimpleTokenizer
from dataset import LMWindowDataset
from model import GPTModel

def plot_attention_heatmap(attention_matrix, step):
    """Create a heatmap visualization of attention matrix"""
    plt.figure(figsize=(10, 8))
    
    # Convert to numpy if tensor
    if torch.is_tensor(attention_matrix):
        attn_np = attention_matrix.numpy()
    else:
        attn_np = attention_matrix
        
    # Create heatmap
    sns.heatmap(attn_np, cmap='Blues', cbar=True, square=True, 
                xticklabels=False, yticklabels=False)
    plt.title(f'Attention Heatmap (Step {step})')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    
    # Add diagonal line to show self-attention pattern
    plt.plot([0, attn_np.shape[1]], [0, attn_np.shape[0]], 'r--', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    return plt.gcf()

def compute_attention_entropy(attention_weights):
    """Compute entropy of attention weights to measure focus/dispersion"""
    # attention_weights: (n_heads, seq_len, seq_len)
    # Add small epsilon to avoid log(0)
    eps = 1e-8
    attention_weights = attention_weights + eps
    
    # Compute entropy along the last dimension (key dimension)
    entropy = -torch.sum(attention_weights * torch.log(attention_weights), dim=-1)
    return entropy  # (n_heads, seq_len)

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
    """Improved learning rate schedule with cosine decay after warmup"""
    if step < args.warmup_steps:
        return args.lr * (step / max(1, args.warmup_steps))
    else:
        # Cosine decay after warmup
        decay_steps = args.max_steps - args.warmup_steps
        cosine_decay = 0.5 * (1 + torch.cos(torch.tensor(torch.pi * (step - args.warmup_steps) / max(1, decay_steps))))
        return args.lr * cosine_decay * 0.1 + args.lr * 0.1  # minimum 10% of original LR

def clip_gradients(model, max_norm):
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

def train(args):
    device = args.device
    os.makedirs(args.out_dir, exist_ok=True)

    # -- pick dataset --
    if args.use_small:
        print("Loading WikiText-2 for quick testing...")
        ds = load_dataset("wikitext", "wikitext-2-raw-v1")
        # Filter out empty lines and join
        train_texts = [text.strip() for text in ds["train"]["text"] if text.strip()]
        text = " ".join(train_texts[:args.small_lines])
    else:
        print("Loading WikiText-103 from Salesforce/wikitext...")
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
        # Filter out empty lines for better quality
        train_texts = [text.strip() for text in ds["train"]["text"] if text.strip()]
        print(f"Found {len(train_texts)} non-empty text segments")
        # Use subset if dataset is very large for efficiency
        if len(train_texts) > args.max_text_segments:
            train_texts = train_texts[:args.max_text_segments]
            print(f"Using first {args.max_text_segments} segments for efficiency")
        text = " ".join(train_texts)

    print(f"Dataset characters: {len(text):,}")

    # Build vocabulary efficiently
    tokenizer = SimpleTokenizer()
    print("Building vocabulary...")
    tokenizer.build_vocab([text], vocab_size=args.vocab_size, min_freq=args.min_token_freq)
    tokenizer.save(os.path.join(args.out_dir, "tokenizer.json"))
    pad_index = tokenizer.stoi[tokenizer.pad_token]
    print(f"Built vocabulary with {len(tokenizer.stoi)} tokens")

    # Tokenize efficiently
    print("Tokenizing text...")
    token_ids = tokenizer.encode(text)
    print(f"Total tokens: {len(token_ids):,}")
    
    # Limit token count for memory efficiency if needed
    if args.max_tokens > 0 and len(token_ids) > args.max_tokens:
        token_ids = token_ids[:args.max_tokens]
        print(f"Truncated to {args.max_tokens} tokens for memory efficiency")

    dataset = LMWindowDataset(token_ids, block_size=args.block_size, pad_index=pad_index, stride=args.stride)
    # Use more workers for faster data loading
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=min(4, args.batch_size),
        pin_memory=device.startswith("cuda"),
        persistent_workers=True if min(4, args.batch_size) > 0 else False
    )

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

    # Compile model for better performance (PyTorch 2.0+)
    if hasattr(torch, 'compile') and device.startswith("cuda"):
        try:
            print("Compiling model for better performance...")
            model = torch.compile(model)
        except Exception as e:
            print(f"Model compilation failed: {e}, continuing without compilation")

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
                
                # Log model parameters stats
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                writer.add_scalar("model/total_params", total_params, global_step)
                writer.add_scalar("model/trainable_params", trainable_params, global_step)
                
                # Log gradient norms for monitoring training stability
                total_grad_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        total_grad_norm += p.grad.data.norm(2).item() ** 2
                total_grad_norm = total_grad_norm ** 0.5
                writer.add_scalar("train/grad_norm", total_grad_norm, global_step)
                
                print(f"[{epoch}] step {global_step} loss {loss.item():.4f} grad_norm {total_grad_norm:.4f}")

            # Log attention maps for visualization
            if global_step % args.tb_attn_interval == 0 and attn_maps is not None:
                try:
                    # Log attention patterns from the specified layer
                    layer_idx = min(args.visualize_layer, len(attn_maps) - 1)
                    if layer_idx < len(attn_maps):
                        attn = attn_maps[layer_idx]  # (batch, n_heads, seq_len, seq_len)
                        
                        # Take first sample and average across heads for visualization
                        attn_avg = attn[0].mean(dim=0).detach().cpu()  # (seq_len, seq_len)
                        
                        # Log as image (attention heatmap)
                        writer.add_figure(f"attention/layer_{layer_idx}_avg", 
                                        plot_attention_heatmap(attn_avg, global_step), 
                                        global_step)
                        
                        # Log individual attention heads
                        for head_idx in range(min(4, attn.size(1))):  # Log first 4 heads
                            head_attn = attn[0, head_idx].detach().cpu()
                            writer.add_figure(f"attention/layer_{layer_idx}_head_{head_idx}", 
                                            plot_attention_heatmap(head_attn, global_step), 
                                            global_step)
                        
                        # Log attention statistics
                        attn_entropy = compute_attention_entropy(attn[0])  # (n_heads, seq_len)
                        writer.add_scalar("attention/avg_entropy", attn_entropy.mean().item(), global_step)
                        writer.add_scalar("attention/max_entropy", attn_entropy.max().item(), global_step)
                        writer.add_scalar("attention/min_entropy", attn_entropy.min().item(), global_step)
                        
                except Exception as e:
                    print(f"Warning: Failed to log attention maps: {e}")

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
        
        # Log additional epoch metrics
        writer.add_scalar("train/tokens_per_second", len(dataset) * args.block_size / (time.time() - t0), epoch)
        writer.add_scalar("train/samples_per_second", len(dataset) / (time.time() - t0), epoch)
        
        # Log memory usage if CUDA available
        if device.startswith("cuda"):
            memory_allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved(device) / 1024**3   # GB
            memory_max = torch.cuda.max_memory_allocated(device) / 1024**3   # GB
            
            writer.add_scalar("system/gpu_memory_allocated_gb", memory_allocated, epoch)
            writer.add_scalar("system/gpu_memory_reserved_gb", memory_reserved, epoch)
            writer.add_scalar("system/gpu_memory_max_gb", memory_max, epoch)
            
            print(f"GPU Memory: {memory_allocated:.1f}GB allocated, {memory_reserved:.1f}GB reserved")
        
        print(f"Epoch {epoch} finished in {time.time()-t0:.1f}s avg_loss {avg_loss:.4f}")
        
        # Calculate and log perplexity
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        writer.add_scalar("train/perplexity", perplexity, epoch)
        print(f"Perplexity: {perplexity:.2f}")

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
    parser.add_argument("--max_text_segments", type=int, default=50000, help="Maximum text segments to use from WikiText")
    parser.add_argument("--max_tokens", type=int, default=0, help="Maximum tokens to use (0 = no limit)")
    parser.add_argument("--min_token_freq", type=int, default=2, help="Minimum frequency for token to be included in vocab")
    parser.add_argument("--block_size", type=int, default=256, help="Sequence length")
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--d_model", type=int, default=512, help="Model dimension")
    parser.add_argument("--d_ff", type=int, default=2048, help="Feed-forward dimension") 
    parser.add_argument("--d_k", type=int, default=64)
    parser.add_argument("--d_v", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--n_layers", type=int, default=8, help="Number of transformer layers")
    parser.add_argument("--vocab_size", type=int, default=8000, help="Vocabulary size (increased for WikiText)")
    parser.add_argument("--batch_size", type=int, default=24, help="Batch size")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Warmup steps (increased)")
    parser.add_argument("--max_steps", type=int, default=10000, help="Maximum training steps for cosine decay")
    parser.add_argument("--gradient_clip", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--tb_attn_interval", type=int, default=200)
    parser.add_argument("--sample_interval", type=int, default=500)
    parser.add_argument("--sample_prompt", type=str, default="The history of artificial intelligence")
    parser.add_argument("--sample_tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.8, help="Lower temperature for better quality")
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--visualize_layer", type=int, default=0)
    parser.add_argument("--early_steps", type=int, default=0, help="If >0 stop after this many steps (debug)")
    args = parser.parse_args()

    # default stride: half block if not specified
    if args.stride is None:
        args.stride = max(1, args.block_size // 2)

    train(args)
