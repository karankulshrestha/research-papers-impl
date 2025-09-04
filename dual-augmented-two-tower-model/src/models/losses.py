import torch
import torch.nn.functional as F

# Binary cross entropy with logits
bce = torch.nn.BCEWithLogitsLoss()

def prediction_loss(scores, labels):
    """
        Calculate the BCE with Logits for scores and labels
        SCORES: Raw similarity scores from the model for user-item interaction
        LABELS: Truth values 1 for positive and 0 for negative 
    """

    return bce(scores, labels)



def mimic_losses(a_u_batch, a_v_batch, p_u, p_v, pos_mask):
    device = p_u.device
    if pos_mask.sum() == 0:
        # If no positives in batch, return 0 loss
        return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
    

    # select only positive batches
    au_pos = a_u_batch[pos_mask]
    av_pos = a_v_batch[pos_mask]

    # detach to prevent the gradient flowing in p_v or p_u
    pu_pos = p_u[pos_mask].detach()
    pv_pos = p_v[pos_mask].detach() 


    # calculation of MSE losses equivalent to Adaptive Mimic Mechanism
    loss_u = F.mse_loss(au_pos, pu_pos)
    loss_v = F.mse_loss(av_pos, pv_pos)

    return loss_u, loss_v



def category_alignment_loss(p_v_batch, categories_batch, eps=1e-8):
    """
    Category Alignment Loss (CAL).
    - Compute covariance of item embeddings per category
    - Align each category's covariance with the majority category
    - Frobenius norm between covariance matrices
    """
    device = p_v_batch.device  # Just gets the hardware (CPU/GPU) for tensors.
    uniq = torch.unique(categories_batch)  # Finds unique category IDs, e.g., [0, 1, 2].
    covs = {}  # Dictionary to store covariance matrices for each category.

    # Compute covariance per category
    for c in uniq:  # Loop over each unique category.
        mask = categories_batch == c  # True/False list: which items are in this category?
        X = p_v_batch[mask]  # Grab embeddings for those items.
        n = X.shape[0]  # Number of items in this category.
        if n < 2:  # Need at least 2 for covariance (can't measure variation with 1).
            continue
        Xc = X - X.mean(dim=0, keepdim=True)  # Subtract mean: center the data.
        cov = (Xc.t() @ Xc) / (n - 1 + eps)  # Compute covariance: matrix math.
        covs[int(c.item())] = cov  # Store it, using category ID as key.

    if len(covs) <= 1:  # If 0 or 1 category, no alignment needed.
        return torch.tensor(0.0, device=device)

    # Pick majority category = category with most samples
    majority = max(covs.keys(), key=lambda k: (categories_batch == k).sum().item())  # Finds the category with the highest count in the batch.
    ref = covs[majority]  # Reference covariance.

    # Compute alignment loss against all other categories
    loss = torch.tensor(0.0, device=device)
    for k, cov in covs.items():
        if k == majority:  # Skip the majority itself.
            continue
        loss = loss + torch.norm(ref - cov, p='fro') ** 2  # Add squared Frobenius distance.

    # Average over categories
    loss = loss / max(1, len(covs) - 1)  # Divide by number of other categories.
    return loss