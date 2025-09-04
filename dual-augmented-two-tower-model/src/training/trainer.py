import torch
from tqdm import tqdm
from src.models.losses import prediction_loss, mimic_losses, category_alignment_loss
from src.data.loader import collate_fn
import os


def train_epoch(model, dataloader, optimizer, device, item_genres_np, lambda_u=0.5, lambda_v=0.5, lambda_ca=1.0, clip_grad=5.0):
    model.train()
    stats = {"pred":0.0, "u":0.0, "v":0.0, "ca":0.0, "total":0.0, "steps":0}
    pbar = tqdm(dataloader, desc="Train")

    for users, items, labels in pbar:
        users = users.to(device); items = items.to(device); labels = labels.float().to(device)
        optimizer.zero_grad()
        scores, p_u, p_v = model(users, items)
        l_pred = prediction_loss(scores, labels)
        a_u_batch = model.a_u(users)
        a_v_batch = model.a_v(items)
        pos_mask = (labels == 1)
        l_u, l_v = mimic_losses(a_u_batch, a_v_batch, p_u, p_v, pos_mask)
        cats = torch.LongTensor(item_genres_np[items.cpu().numpy()]).to(device)
        l_ca = category_alignment_loss(p_v, cats)

        loss = l_pred + lambda_u * l_u + lambda_v * l_v + lambda_ca * l_ca

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        optimizer.step()

        stats["pred"] += l_pred.item()
        stats["u"] += (l_u.item() if isinstance(l_u, torch.Tensor) else float(l_u))
        stats["v"] += (l_v.item() if isinstance(l_v, torch.Tensor) else float(l_v))
        stats["ca"] += l_ca.item()
        stats["total"] += loss.item()
        stats["steps"] += 1

        pbar.set_postfix({
            "L_pred": f"{stats['pred']/stats['steps']:.4f}",
            "L_u": f"{stats['u']/stats['steps']:.4f}",
            "L_v": f"{stats['v']/stats['steps']:.4f}",
            "L_ca": f"{stats['ca']/stats['steps']:.4f}",
            "L_all": f"{stats['total']/stats['steps']:.4f}",
        })
    
    return {k:(v / stats["steps"] if k != "steps" else v) for k,v in stats.items()}



def save_checkpoint(path, model, optimizer, epoch, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "opt_state": optimizer.state_dict()
    }
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)