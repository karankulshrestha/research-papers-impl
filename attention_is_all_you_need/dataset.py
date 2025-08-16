# dataset.py
import torch
from torch.utils.data import Dataset

class LMWindowDataset(Dataset):
    """
    Produces (x, y) pairs for autoregressive LM training.
    Each sample: x (block_size tokens), y (block_size tokens shifted by 1)
    Pads the last window with pad_index when needed.
    Optimized for memory efficiency and better training coverage.
    """

    def __init__(self, token_ids: list, block_size: int, pad_index: int, stride: int = None):
        self.tokens = token_ids
        self.block_size = block_size
        self.pad_index = pad_index
        # Use smaller stride for better data coverage
        self.stride = stride if stride is not None else max(1, block_size // 4)
        self.windows = []
        n = len(self.tokens)

        if n <= 1:
            pad_needed = block_size + 1
            w = [self.pad_index] * pad_needed
            self.windows.append(w)
        else:
            # Create overlapping windows for better training
            for i in range(0, max(1, n - block_size), self.stride):
                chunk = self.tokens[i:i + block_size + 1]
                if len(chunk) < block_size + 1:
                    # Pad the last chunk if needed
                    chunk = chunk + [self.pad_index] * (block_size + 1 - len(chunk))
                self.windows.append(chunk)
            
            # Ensure we have at least one complete window
            if len(self.windows) == 0:
                chunk = self.tokens[:block_size + 1]
                if len(chunk) < block_size + 1:
                    chunk = chunk + [self.pad_index] * (block_size + 1 - len(chunk))
                self.windows.append(chunk)

        print(f"Created {len(self.windows)} training windows from {n} tokens with stride {self.stride}")
        print(f"Data coverage: {len(self.windows) * self.stride / max(1, n) * 100:.1f}%")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        w = self.windows[idx]
        x = torch.tensor(w[:self.block_size], dtype=torch.long)
        y = torch.tensor(w[1:self.block_size + 1], dtype=torch.long)
        return x, y
