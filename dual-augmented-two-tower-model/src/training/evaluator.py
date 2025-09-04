import numpy as np
import torch
import math


def compute_metrics(item_embs_cpu, model, test_pairs, train_map, device, K_list=[10, 50, 100]):
    """
    item_embs_cpu: numpy array (num_items, dim)
    test_pairs: numpy array (n,2) or list of tuples
    train_map: dict user -> set(seen_items)
    returns: hr dict, mrr float, ndcg dict
    """
    num_items = item_embs_cpu.shape[0]
    ks = {k: 0 for k in K_list}     # hit counts for HR@K
    mrr_sum = 0.0                   # sum of reciprocal ranks
    ndcg_sum = {k: 0.0 for k in K_list}  # sum of NDCG@K
    tot = len(test_pairs)

    model.eval()
    with torch.no_grad():
        for (u, test_item) in test_pairs:
            u = int(u); test_item = int(test_item)

            # Build user embedding
            uid = torch.LongTensor([u]).to(next(model.parameters()).device)
            eu = model.u_emb(uid); au = model.a_u(uid)
            zu = torch.cat([eu, au], dim=1)
            pu = model.user_tower(zu)
            pu = torch.nn.functional.normalize(pu, dim=1).cpu().numpy().squeeze(0)

            # Compute similarity scores
            scores = item_embs_cpu @ pu

            # Mask out training items
            seen = train_map.get(u, set())
            if seen:
                scores[list(seen)] = -1e9

            # Rank all items
            order = np.argsort(-scores)

            # Find true test item’s rank
            pos = np.where(order == test_item)[0]
            if len(pos) > 0:
                rank = int(pos[0]) + 1
                mrr_sum += 1.0 / rank

            # Compute HitRate@K
            for K in K_list:
                topk = order[:K]
                if test_item in topk:
                    ks[K] += 1

            # Compute NDCG@K
            for K in K_list:
                # Find rank of test item
                rank = np.where(order == test_item)[0]
                if len(rank) > 0:
                    rank = rank[0] + 1  # 1-based rank
                    if rank <= K:
                        # DCG: relevance / log2(rank+1)
                        dcg = 1.0 / np.log2(rank + 1)
                        # IDCG: ideal DCG (relevant item at position 1)
                        idcg = 1.0 / np.log2(1 + 1)
                        ndcg = dcg / idcg
                        ndcg_sum[K] += ndcg

    # Final averages
    hr = {K: ks[K] / tot for K in ks}
    mrr = mrr_sum / tot
    ndcg = {K: ndcg_sum[K] / tot for K in ndcg_sum}
    return hr, mrr, ndcg