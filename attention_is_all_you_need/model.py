import torch
import math
import torch.nn as nn

class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, dropout, max_len=5000):

        """ Positional Embedding Module"""

        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)

        div_term = torch.arange(0, d_model, 2) * -(math.log(10000) / d_model)

        pe[:, 0::2] = torch.sin(position * torch.exp(div_term))
        pe[:, 1::2] = torch.cos(position * torch.exp(div_term))

        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)
    

    def forward(self, x):
        """ Forward pass of the Positional Embedding Module"""
        """ X -> embedding dimention: (batch_size, seq_len, d_model) """
        
        assert x.size(-1) == self.pe.size(-1), "Embedding dimension of x and pe must match"

        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class ScaledProductAttention(nn.Module):

    def __init__(self, d_k, device):
        super().__init__()
        self.d_k = d_k
        self.device = device
    
    def forward(self, Q, K, V, attention_mask):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, device=Q.device, dtype=Q.dtype))
        attention_mask = torch.as_tensor(attention_mask, dtype=torch.bool, device=scores.device)
        scores.masked_fill_(attention_mask, float('-inf'))
        scores = torch.softmax(scores, dim=-1)
        attention_weights = scores @ V

        return attention_weights, scores
        


class MultiheadAttention(nn.Module):

    def __init__(self, d_model, n_heads, d_k, d_v, device):
        super().__init__()
        self.WQ = nn.Linear(d_model, n_heads * d_k)
        self.WK = nn.Linear(d_model, n_heads * d_k)
        
        self.WV = nn.Linear(n_heads * d_v, d_model)

        self.linear = nn.Linear(n_heads * d_v, d_model)

        self.layerNorm = nn.LayerNorm(d_model)
        self.device = device

        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
    
    def forward(self, Q, K, V, attention_mask):
        batch_size = Q.size(0)

        # q_s, k_s, v_s shapes: (batch_size, n_heads, seq_len, d_k or d_v)

        q_s = self.WQ(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.WK(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_s = self.WV(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        # Original mask: (batch_size, seq_len, seq_len)
        # .unsqueeze(1): (batch_size, 1, seq_len, seq_len)
        # .repeat(1, n_heads, 1, 1): (batch_size, n_heads, seq_len, seq_len)
        self.attention_mask = attention_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)

        # attention_weights shape: (batch_size, n_heads, seq_len, d_k)
        # scores shape: (batch_size, n_heads, seq_len, d_k)
        attention_weights, scores = ScaledProductAttention(self.d_k, self.device).forward(q_s, k_s, v_s, self.attention_mask)

        scores = scores.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k) 

        scores = self.linear(scores)

        return self.layerNorm(scores + Q), attention_weights


        

