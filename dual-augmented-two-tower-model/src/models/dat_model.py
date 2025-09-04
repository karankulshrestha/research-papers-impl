from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """
    Simple Multi Layers used in two tower user tower and item tower
    """
    def __init__(self, input_dim, hidden_dims=[256, 128, 32]):
        super().__init__()
        layers = []
        prev = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU(inplace=True))
            prev = h
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)



class DATModel(nn.Module):
    """
    Dual Augmented Two-Tower (DAT) model.
    - one tower for users, one for items
    - Both towers receive concatenation of (embedding + augmented embedding)
    """

    def __init__(self, num_users, num_items, emb_dim=32, aug_dim=32, tower_hidden=[256, 128, 32]):
        super().__init__()

        # base embeddings u and v as per the paper
        self.u_emb = nn.Embedding(num_users, emb_dim)
        self.v_emb = nn.Embedding(num_items, emb_dim)

        # augmented embeddings (carry cross tower information as per the paper)
        self.a_u = nn.Embedding(num_users, aug_dim)
        self.a_v = nn.Embedding(num_items, aug_dim)

        # Tower structure processing: (concatenated embedding)
        self.user_tower = MLP(emb_dim + aug_dim, tower_hidden)
        self.item_tower = MLP(emb_dim + aug_dim, tower_hidden)

        
        # Initialize embeddings with small random values (normal distribution) to prevent vanishing/exploding of the gradients
        # and making training stable
        nn.init.normal_(self.u_emb.weight, 0, 0.01)
        nn.init.normal_(self.v_emb.weight, 0, 0.01)
        nn.init.normal_(self.a_u.weight, 0, 0.01)
        nn.init.normal_(self.a_v.weight, 0, 0.01)
    

    def forward(self, u_idx, v_idx):
        # looking up base embeddings
        eu = self.u_emb(u_idx) # u_idx = torch.tensor[id1, id2, id3,..] -> lookup corresponding in self.u_emb -> fetch (batch, emb) and emb = 32
        ev = self.v_emb(v_idx) # v_idx = torch.tensor[id1, id2, id3,..] -> lookup corresponding in self.v_emb -> fetch (batch, emb) and emb = 32


        # looking up augmented embeddings
        au = self.a_u(u_idx) # (batch, emb)
        av = self.a_v(v_idx) # (batch, emb)

        zu = torch.cat([eu, au], dim=1) # input to user tower
        zv = torch.cat([ev, av], dim=1) # input to item tower

        # pass through towers
        pu = self.user_tower(zu) # (batch, 32)
        pv = self.item_tower(zv) # (batch, 32)

        # Apply L2 Normalization for consine similarity
        pu = F.normalize(pu, dim=1) # (batch, 32)
        pv = F.normalize(pv, dim=1) # (batch, 32)

        # final score = dot product of user tower output and item tower output
        scores = (pu * pv).sum(dim=1) # (batch,) -> one score per user-item 

        return scores, pu, pv
    

    def item_embeddings(self, device='cpu', batch_size=1024):
        """
        compute and return final item embeddings p_v for all the items.
        Useful for retrieval after training
        """

        self.eval()
        embs = []
        N = self.v_emb.num_embeddings # number of items
        with torch.no_grad():
            for s in range(0, N, batch_size):
                idx = torch.arange(s, min(N, s + batch_size), device=device, dtype=torch.long)
                ev = self.v_emb(idx)
                av = self.a_v(idx)
                zv = torch.cat([ev, av], dim=1)
                pv = self.item_tower(zv)
                pv = F.normalize(pv, dim=1)
                embs.append(pv.cpu())
        
        return torch.concat(embs, dim=0)