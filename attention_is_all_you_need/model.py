# model.py  (converted from your gpt_model.py, fixed small bugs)
import torch
import math
import torch.nn as nn

def get_attn_pad_mask(seq_q, seq_k, pad_index):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.eq(pad_index).unsqueeze(1)   # (batch, 1, len_k)
    pad_attn_mask = pad_attn_mask.expand(batch_size, len_q, len_k)
    return pad_attn_mask  # boolean tensor

def get_attn_subsequent_mask(seq):
    batch_size, seq_len = seq.size()
    subsequent_mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool, device=seq.device), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(batch_size, seq_len, seq_len)
    return subsequent_mask

class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        # build div_term tensor (device agnostic)
        div_term = torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model)
        div_term = torch.exp(div_term)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        assert x.size(-1) == self.pe.size(-1), "Embedding dimension mismatch"
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class ScaledProductAttention(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, attention_mask):
        # Q: (batch, n_heads, seq_q, d_k)
        # K: (batch, n_heads, seq_k, d_k)
        # V: (batch, n_heads, seq_k, d_v)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=Q.dtype, device=Q.device))
        # attention_mask: expected boolean mask of shape (batch, n_heads, seq_q, seq_k)
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask, float('-inf'))
        scores = torch.softmax(scores, dim=-1)
        attention_output = scores @ V  # (batch, n_heads, seq_q, d_v)
        return attention_output, scores # shape: (batch, n_heads, seq_q, d_v), (batch, n_heads, seq_q, seq_k)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v):
        super().__init__()
        self.WQ = nn.Linear(d_model, n_heads * d_k)
        self.WK = nn.Linear(d_model, n_heads * d_k)
        self.WV = nn.Linear(d_model, n_heads * d_v)
        self.linear = nn.Linear(n_heads * d_v, d_model)
        self.layerNorm = nn.LayerNorm(d_model)

        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v

    def forward(self, Q, K, V, attention_mask): # shape: (batch, seq_len, d_model), (batch, seq_len, d_model), (batch, seq_len, d_model), (batch, seq_len, seq_len)
        # Q, K, V: (batch, seq_len, d_model)
        batch_size = Q.size(0)
        q_s = self.WQ(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # (batch, seq_len, d_model) -> (batch, seq_len, n_heads, d_k) -> (batch, n_heads, seq_len, d_k)
        k_s = self.WK(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_s = self.WV(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        # attention_mask: (batch, seq_q, seq_k) -> expand to (batch, n_heads, seq_q, seq_k)
        attn_mask = None
        if attention_mask is not None:
            # shape: (batch, seq_q, seq_k) -> (batch, 1, seq_q, seq_k) -> (batch, n_heads, seq_q, seq_k)
            attn_mask = attention_mask.unsqueeze(1).expand(batch_size, self.n_heads, attention_mask.size(1), attention_mask.size(2))

        attention_output, attention_scores = ScaledProductAttention(self.d_k).forward(q_s, k_s, v_s, attn_mask)
        concat_heads = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)
        output = self.linear(concat_heads)
        return self.layerNorm(output + Q), attention_scores

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.l1 = nn.Linear(d_model, d_ff)
        self.l2 = nn.Linear(d_ff, d_model)
        self.act = GELU()
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        residual = inputs
        output = self.l1(inputs)
        output = self.act(output)
        output = self.l2(output)
        return self.layer_norm(output + residual)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, d_k, d_v, n_heads):
        super().__init__()
        self.enc_self_attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v)
        self.pos_ffn = PoswiseFeedForwardNet(d_model=d_model, d_ff=d_ff)

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, d_ff, d_k, d_v, n_heads, n_layers, pad_index, device):
        super().__init__()
        self.device = device
        self.pad_index = pad_index
        self.src_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model=d_model, dropout=0.0)
        layers = []
        for _ in range(n_layers):
            layers.append(EncoderLayer(d_model=d_model, d_ff=d_ff, d_k=d_k, d_v=d_v, n_heads=n_heads))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        enc_outputs = self.src_emb(x)
        enc_outputs = self.pos_emb(enc_outputs)
        enc_self_attn_mask = get_attn_pad_mask(x, x, self.pad_index)  # bool mask
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        enc_self_attns = torch.stack(enc_self_attns)  # (n_layers, batch, n_heads, seq, seq)
        enc_self_attns = enc_self_attns.permute([1, 0, 2, 3, 4])  # (batch, n_layers, n_heads, seq, seq)
        return enc_outputs, enc_self_attns

class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, d_k, d_v, n_heads):
        super().__init__()
        self.dec_self_attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v)
        self.dec_enc_attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v)
        self.pos_ffn = PoswiseFeedForwardNet(d_model=d_model, d_ff=d_ff)

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, d_ff, d_k, d_v, n_heads, n_layers, pad_index, device):
        super().__init__()
        self.pad_index = pad_index
        self.device = device
        self.tgt_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model=d_model, dropout=0.0)
        layers = []
        for _ in range(n_layers):
            layers.append(DecoderLayer(d_model=d_model, d_ff=d_ff, d_k=d_k, d_v=d_v, n_heads=n_heads))
        self.layers = nn.ModuleList(layers)

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        dec_outputs = self.tgt_emb(dec_inputs)
        dec_outputs = self.pos_emb(dec_outputs)
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs, self.pad_index)
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)
        dec_self_attn_mask = dec_self_attn_pad_mask | dec_self_attn_subsequent_mask
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs, self.pad_index)

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(
                dec_inputs=dec_outputs,
                enc_outputs=enc_outputs,
                dec_self_attn_mask=dec_self_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)

        dec_self_attns = torch.stack(dec_self_attns).permute([1, 0, 2, 3, 4])
        dec_enc_attns = torch.stack(dec_enc_attns).permute([1, 0, 2, 3, 4])
        return dec_outputs, dec_self_attns, dec_enc_attns

class MaskedDecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, d_k, d_v, n_heads):
        super().__init__()
        self.dec_self_attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v)
        self.pos_ffn = PoswiseFeedForwardNet(d_model=d_model, d_ff=d_ff)

    def forward(self, dec_inputs, dec_self_attn_mask): # input shapes: (batch_size, seq_len, d_model), (batch_size, seq_len, seq_len)
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn

class MaskedDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, d_ff, d_k, d_v, n_heads, n_layers, pad_index, device):
        super().__init__()
        self.pad_index = pad_index
        self.tgt_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model=d_model, dropout=0.0)
        layers = []
        for _ in range(n_layers):
            layers.append(MaskedDecoderLayer(d_model=d_model, d_ff=d_ff, d_k=d_k, d_v=d_v, n_heads=n_heads))
        self.layers = nn.ModuleList(layers)

    def forward(self, dec_inputs):
        dec_outputs = self.tgt_emb(dec_inputs) # shape: (batch_size, seq_len) -> (batch_size, seq_len, d_model)
        dec_outputs = self.pos_emb(dec_outputs) # shape: (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs, self.pad_index)
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)
        dec_self_attn_mask = dec_self_attn_pad_mask | dec_self_attn_subsequent_mask # shape: (batch_size, seq_len, seq_len)
        dec_self_attns = []
        for layer in self.layers:
            dec_outputs, dec_self_attn = layer(dec_outputs, dec_self_attn_mask)
            dec_self_attns.append(dec_self_attn)
        
        # torch.stack -> stack up all the layers output dec_self_attn and add new dimension (n_layer)

        # (n_layers, batch_size, n_heads, seq_len, seq_len) -> (batch_size, n_layers, n_heads, seq_len, seq_len) 
        dec_self_attns = torch.stack(dec_self_attns).permute([1, 0, 2, 3, 4]) # transform the dimension 1 and 0
        return dec_outputs, dec_self_attns

class GPTModel(nn.Module):
    def __init__(self, vocab_size, d_model, d_ff, d_k, d_v, n_heads, n_layers, pad_index, device):
        super().__init__()
        self.decoder = MaskedDecoder(
            vocab_size=vocab_size,
            d_model=d_model, d_ff=d_ff,
            d_k=d_k, d_v=d_v, n_heads=n_heads,
            n_layers=n_layers, pad_index=pad_index,
            device=device)
        self.projection = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, dec_inputs):
        dec_outputs, dec_self_attns = self.decoder(dec_inputs)
        dec_logits = self.projection(dec_outputs)
        return dec_logits, dec_self_attns
