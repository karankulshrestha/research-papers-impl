import os
import zipfile
from PIL.Image import item
import requests
import numpy as np
import pandas as pd
import json
from collections import defaultdict

RAW_DIR = "data/raw/ml-100k"
PROC_DIR = "data/processed"


ML100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"

def download_and_extract(dest_dir="data/raw"):
    os.makedirs(dest_dir, exist_ok=True)
    zip_path = os.path.join(dest_dir, "ml-100k.zip")
    
    if not os.path.exists(zip_path):
        print("Download movie-lens-100k...")
        r = requests.get(ML100K_URL, stream=True)

        with open(zip_path, 'wb') as f:
            for chunk in r.iter_content(1024 * 1024):
                f.write(chunk)
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(dest_dir)
    
    print("downloading and extracted to", dest_dir)



def load_raw(raw_dir=RAW_DIR):

    udata = os.path.join(raw_dir, "u.data")
    uitem = os.path.join(raw_dir, "u.item")

    if not os.path.exists(udata) or not os.path.exists(uitem):
        raise FileNotFoundError("Extected u.data and u.item in ", raw_dir)
    
    ratings = pd.read_csv(udata, sep='\t', names=['user_raw', 'item_raw', 'rating', 'ts'], engine='python')

    items = pd.read_csv(uitem, sep='|', names=['movie_id','title','release_date','video_release','imdb_url'] + [f'genre_{i}' for i in range(19)], encoding='latin-1', engine='python')

    return ratings, items



def build_maps(ratings):
    unique_users = sorted(ratings['user_raw'].unique())
    unique_items = sorted(ratings['item_raw'].unique())

    user_map = {u: idx for idx, u in enumerate(unique_users)}
    item_map = {v: idx for idx, v in enumerate(unique_items)}

    return user_map, item_map


def remap_and_save(ratings, items, user_map, item_map):
    ratings['user_idx'] = ratings['user_raw'].map(user_map)
    ratings['item_idx'] = ratings['item_raw'].map(item_map)

    print(ratings)

    genre_cols = [c for c in items.columns if c.startswith('genre_')]


    item_genres = [] # this contains the primary genre for each movie of total 1682
    for _, row in items.iterrows():
        flags = row[genre_cols].values.astype(int)
        idxs = np.where(flags == 1)[0]
        primary = int(idxs[0]) if len(idxs) > 0 else 0
        item_genres.append(primary)
    
    movieid_to_idx = {row['movie_id']: item_map[row['movie_id']] for _, row in items.iterrows() if row['movie_id'] in item_map}
    # Make sure ordering matches mapped index order
    num_items = len(item_map)
    item_genres_arr = np.zeros(num_items, dtype=np.int32)

    for movie_id, mapped_idx in movieid_to_idx.items():
        row = items[items['movie_id'] == movie_id].iloc[0]
        flags = row[genre_cols].values.astype(int)
        idxs = np.where(flags == 1)[0]
        item_genres_arr[mapped_idx] = int(idxs[0] if len(idxs) > 0 else 0)
    
    os.makedirs(PROC_DIR, exist_ok=True)

    np.savez_compressed(os.path.join(PROC_DIR, 'interactions.npz'),
                        user_idx=ratings['user_idx'].values.astype(np.int32),
                        item_idx=ratings['item_idx'].values.astype(np.int32),
                        rating=ratings['rating'].values.astype(np.int8),
                        ts=ratings['ts'].values.astype(np.int64))
    
    np.save(os.path.join(PROC_DIR, 'item_genres.npy'), item_genres_arr)

    inv_user = [None] * len(user_map)
    for k, v in user_map.items():
        inv_user[v] = k
    
    inv_item = [None] * len(item_map)
    for k, v in item_map.items():
        inv_item[v] = k
    
    np.savez_compressed(os.path.join(PROC_DIR, 'mapping.npz'), users=inv_user, items=inv_item)
    print("Saved processed files to", PROC_DIR)

    return ratings, item_genres_arr
    


def leave_one_out_split(ratings):
    df = ratings.sort_values(['user_idx', 'ts'])
    train_rows = []
    val_rows = []
    test_rows = []
    user_pos = defaultdict(list)
    for user, g in df.groupby('user_idx'):
        items = list(g['item_idx'].values)
        if len(items) == 1:
            train_rows.append((user, items[0]))
        elif len(items) == 2:
            train_rows.append((user, items[0]))
            test_rows.append((user, items[1]))
        else:
            train_rows.extend([(user, item) for item in items[:-2]])
            val_rows.append((user, items[-2]))
            test_rows.append((user, items[-1]))
        
        for item in items[:-1]:
            user_pos[user].append(item)
    
    # convert to numpy array for saving
    def to_np(list_of_pairs):
        if len(list_of_pairs) == 0:
            return np.empty((0, 2), dtype=np.int32)
        
        return np.array(list_of_pairs, dtype=np.int32)
    
    os.makedirs(PROC_DIR, exist_ok=True)
    np.savez_compressed(os.path.join(PROC_DIR, 'splits.npz'),
    train=to_np(train_rows), val=to_np(val_rows), test=to_np(test_rows))

    # save user_pos map (JSON WITH LISTS)
    user_pos_json = {str(u): [int(x) for x in v] for u, v in user_pos.items()}
    with open(os.path.join(PROC_DIR, 'user_pos.json'), 'w') as f:
        json.dump(user_pos_json, f)
    
    print("Saved splits and user pos to", PROC_DIR)

    return train_rows, val_rows, test_rows, user_pos



def sanity_checks(ratings, train_rows, val_rows, test_rows, item_genres):
    num_users = ratings['user_idx'].nunique()
    num_items = ratings['item_idx'].nunique()
    total_inter = len(ratings)
    
    print("Num users:", num_users)
    print("Num items:", num_items)
    print("Total interactions:", total_inter)
    print("Train interactions:", len(train_rows))
    print("Val interactions:", len(val_rows))
    print("Test interactions:", len(test_rows))
    print("Item genres shape:", item_genres.shape, "unique genres:", np.unique(item_genres).shape[0])



def main(do_download=False):
    if do_download:
        download_and_extract("data/raw")
    ratings, items = load_raw()
    user_map, item_map = build_maps(ratings)
    ratings, item_genres_arr = remap_and_save(ratings, items, user_map, item_map)
    train_rows, val_rows, test_rows, user_pos = leave_one_out_split(ratings)
    sanity_checks(ratings, train_rows, val_rows, test_rows, item_genres_arr)
    print("Done preprocessing.")


if __name__ == "__main__":
    main(do_download=False)



# ratings, items = load_raw()

# print("RATINGS")

# print(ratings)

# print("========================")

# print("ITEMS")

# print(items)


# user_map, item_map = build_maps(ratings)

# print("USER MAP")

# print(user_map)

# print("============================")

# print("ITEM MAP")

# print(item_map)
