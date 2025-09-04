import numpy as np
import os

PROC_DIR = "data/processed"

print("=== ANALYZING PROCESSED DATA FILES (excluding JSON) ===\n")

# 1. interactions.npz - User-item interactions
print("1. interactions.npz")
print("-" * 50)
inter = np.load(os.path.join(PROC_DIR, "interactions.npz"))
print(f"Keys: {list(inter.keys())}")
print(f"Total interactions: {len(inter['user_idx'])}")
print("Data types:")
print(f"  user_idx: {inter['user_idx'].dtype}, shape: {inter['user_idx'].shape}")
print(f"  item_idx: {inter['item_idx'].dtype}, shape: {inter['item_idx'].shape}")
print(f"  rating: {inter['rating'].dtype}, shape: {inter['rating'].shape}")
print(f"  ts: {inter['ts'].dtype}, shape: {inter['ts'].shape}")
print("\nFirst 5 interactions:")
for i in range(5):
    print(f"  User {inter['user_idx'][i]}, Item {inter['item_idx'][i]}, Rating {inter['rating'][i]}, Timestamp {inter['ts'][i]}")
print()

# 2. item_genres.npy - Genre information for each item
print("2. item_genres.npy")
print("-" * 50)
item_genres = np.load(os.path.join(PROC_DIR, "item_genres.npy"))
print(f"Shape: {item_genres.shape}")
print(f"Data type: {item_genres.dtype}")
print(f"Unique genres: {len(np.unique(item_genres))}")
print(f"Genre distribution: {np.bincount(item_genres)}")
print("First 10 item genres:")
print(f"  {item_genres[:10].tolist()}")
print()

# 3. mapping.npz - ID mappings
print("3. mapping.npz")
print("-" * 50)
maps = np.load(os.path.join(PROC_DIR, "mapping.npz"), allow_pickle=True)
print(f"Keys: {list(maps.keys())}")
print(f"Number of users: {len(maps['users'])}")
print(f"Number of items: {len(maps['items'])}")
print("First 5 user mappings (internal_idx -> original_id):")
for i in range(5):
    print(f"  User {i} -> {maps['users'][i]}")
print("First 5 item mappings (internal_idx -> original_movie_id):")
for i in range(5):
    print(f"  Item {i} -> {maps['items'][i]}")
print()

# 4. splits.npz - Train/validation/test splits
print("4. splits.npz")
print("-" * 50)
splits = np.load(os.path.join(PROC_DIR, "splits.npz"))
print(f"Keys: {list(splits.keys())}")
for name in ["train", "val", "test"]:
    arr = splits[name]
    print(f"{name.upper()} set: {arr.shape[0]} interactions, shape: {arr.shape}")
    print(f"  Data type: {arr.dtype}")
    print(f"  First 3 pairs (user_idx, item_idx):")
    for i in range(min(3, len(arr))):
        print(f"    {arr[i].tolist()}")
    print()

print("=== SUMMARY ===")
print(f"Total users: {len(maps['users'])}")
print(f"Total items: {len(maps['items'])}")
print(f"Total interactions: {len(inter['user_idx'])}")
print(f"Training interactions: {splits['train'].shape[0]}")
print(f"Validation interactions: {splits['val'].shape[0]}")
print(f"Test interactions: {splits['test'].shape[0]}")