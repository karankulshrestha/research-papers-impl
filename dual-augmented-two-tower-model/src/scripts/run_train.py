import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader

from src.utils.utils import set_seed, makedirs
from src.data.loader import load_processed, NegDataset, collate_fn
from src.models.dat_model import DATModel
from src.training.trainer import train_epoch, save_checkpoint
from src.training.evaluator import compute_metrics

def build_train_map(train_pairs): 
    m = {}
    for u, v in train_pairs:
        u = int(u); v = int(v)
        m.setdefault(u, set()).add(v)
    return m

def main(cfg):
    set_seed(cfg['seed'])
    data = load_processed(cfg['data_dir'])
    item_genres = data['item_genres']
    splits = data['splits']
    train_pairs = splits['train']
    val_pairs = splits['val']
    test_pairs = splits['test']

    num_users = len(data['mappings']['users'])
    num_items = len(data['mappings']['items'])
    print(f"num_users {num_users} num_items {num_items}")

    train_dataset = NegDataset(train_pairs, num_items, data['user_pos'], neg_samples=cfg['neg_samples'])
    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DATModel(num_users, num_items, emb_dim=cfg['emb_dim'], aug_dim=cfg['aug_dim'],
                     tower_hidden=cfg['tower_hidden']).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    train_map = build_train_map(train_pairs)

    makedirs(cfg['save_dir'])
    for epoch in range(1, cfg['epochs'] + 1):
        stats = train_epoch(model, train_loader, optimizer, device, item_genres,
                            lambda_u=cfg['lambda_u'], lambda_v=cfg['lambda_v'], lambda_ca=cfg['lambda_ca'])
        print(f"Epoch {epoch} train_stats: {stats}")
        # evaluate
        item_embs = model.item_embeddings(device=device).cpu().numpy()
        hr, mrr, ndcg = compute_metrics(item_embs, model, test_pairs, train_map, device, K_list=cfg['eval_k'])
        print(f"Epoch {epoch} EVAL HR@50 {hr.get(50,'NA'):.4f} MRR {mrr:.4f} NDCG@50 {ndcg.get(50,'NA'):.4f}")
        # save
        ckpt_path = os.path.join(cfg['save_dir'], f"dat_epoch{epoch}.pt")
        save_checkpoint(ckpt_path, model, optimizer, epoch, extra={'hr50': hr.get(50, 0.0), 'mrr': mrr})
    print("training finished")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/default.yaml")
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))
    main(cfg)