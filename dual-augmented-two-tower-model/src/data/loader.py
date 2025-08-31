import os
import json
import numpy as np
import random
from torch.utils.data import Dataset

PROC_DIR = "data/processed"

def load_processed(proc_Dir=PROC_DIR):
    interactions = np.load(os.path.join(proc_Dir, "interactions.npz"))
    mappings = np.load(os.path.join(proc_Dir, "mapping.npz"), allow_pickle=True)
    item_genres = np.load(os.path.join(proc_Dir, "item_genres.npy"))
    splits = np.load(os.path.join(proc_Dir, "splits.npz"))
    with open(os.path.join(proc_Dir, "user_pos.json"), "r") as f:
        user_pos = json.load(f)
    
    user_pos = {int(k): set(v) for k, v in user_pos.items()}

    return {
        "interactions": interactions,
        "mappings": mappings,
        "item_genres": item_genres,
        "splits": splits,
        "user_pos": user_pos
    }

# res = load_processed()
# print(res)


class NegDataset(Dataset):
    """
    Positive pairs are provided as numpy array shape (n, 2)
    __getitem__ returns: user(int), pos_item(int), list_of_negatives 
    """

    def __init__(self, pos_pairs, num_items, user_pos_map, neg_samples=4) -> None:
        self.pos = pos_pairs.tolist() if hasattr(pos_pairs, "tolist") else list(pos_pairs)
        self.num_items = int(num_items)
        self.neg = int(neg_samples)
        self.user_pos = {int(u): set(vs) for u, vs in user_pos_map.items()}
    
    def __len__(self):
        return len(self.pos)
    
    def __getitem__(self, idx):
        
        u = int(self.pos[idx][0])
        pos = int(self.pos[idx][1])
        
        negs = []
        while len(negs) < self.neg:
            cand = random.randrange(self.num_items)
            if cand not in self.user_pos.get(u, set()) and cand != pos:
                negs.append(cand)
        
        return u, pos, negs



def collate_fn(batch):
    users = []
    items = []
    labels = []

    for u, pos, negs in batch:
        users.append(u); items.append(pos); labels.append(1)

        for n in negs:
            users.append(u); items.append(n); labels.append(0)
    
    import torch

    return torch.LongTensor(users), torch.LongTensor(items), torch.FloatTensor(labels)