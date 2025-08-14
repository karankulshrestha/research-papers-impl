import torch
import math
import torch.nn as nn
import numpy as np

def get_attn_pad_mask(seq_q, seq_k, pad_index):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(pad_index).unsqueeze(1)
    pad_attn_mask = torch.as_tensor(pad_attn_mask, dtype=torch.int)
    return pad_attn_mask.expand(batch_size, len_q, len_k)

def get_attn_subsequent_mask(seq):
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)
    subsequent_mask = torch.from_numpy(subsequent_mask).int()
    return subsequent_mask



class GELU(nn.Module):

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class PositionalEncoding(nn.Module):

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
        # Q shape: (batch_size, n_heads, seq_len_q, d_k)
        # K shape: (batch_size, n_heads, seq_len_k, d_k)
        # V shape: (batch_size, n_heads, seq_len_k, d_v)

        # 1. Compute raw attention scores
        # scores shape: (batch_size, n_heads, seq_len_q, seq_len_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.d_k, device=Q.device, dtype=Q.dtype)
        )

        # 2. Apply mask
        attention_mask = torch.as_tensor(attention_mask, dtype=torch.bool, device=scores.device)
        scores.masked_fill_(attention_mask, float('-inf'))

        # 3. Softmax over last dimension (keys)
        scores = torch.softmax(scores, dim=-1)  # still (batch_size, n_heads, seq_len_q, seq_len_k)

        # 4. Multiply by V
        # attention_weights shape: (batch_size, n_heads, seq_len_q, d_v)
        attention_weights = scores @ V

        # Return: (output after weighting V, attention map)
        return attention_weights, scores


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v, device):
        super().__init__()
        self.WQ = nn.Linear(d_model, n_heads * d_k)  # projects embedding → Q for all heads
        self.WK = nn.Linear(d_model, n_heads * d_k)  # projects embedding → K for all heads
        self.WV = nn.Linear(d_model, n_heads * d_v)  # projects embedding → V for all heads

        self.linear = nn.Linear(n_heads * d_v, d_model)  # combines heads back to d_model

        self.layerNorm = nn.LayerNorm(d_model)
        self.device = device

        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
    
    def forward(self, Q, K, V, attention_mask):
        # Q, K, V: (batch_size, seq_len, d_model)
        batch_size = Q.size(0)

        # 1. Project to multi-head Q, K, V
        # After WQ/WK/WV: (batch_size, seq_len, n_heads * d_k or d_v)
        # .view(..., n_heads, d_k/d_v): (batch_size, seq_len, n_heads, d_k/d_v)
        # .transpose(1, 2): (batch_size, n_heads, seq_len, d_k/d_v)
        q_s = self.WQ(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.WK(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_s = self.WV(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        # 2. Prepare mask
        # Original: (batch_size, seq_len_q, seq_len_k)
        # .unsqueeze(1): (batch_size, 1, seq_len_q, seq_len_k)
        # .repeat(...): (batch_size, n_heads, seq_len_q, seq_len_k)
        self.attention_mask = attention_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)

        # 3. Scaled dot-product attention for all heads
        # attention_output: (batch_size, n_heads, seq_len_q, d_v)
        # attention_scores: (batch_size, n_heads, seq_len_q, seq_len_k)
        attention_output, attention_scores = ScaledProductAttention(
            self.d_k, self.device
        ).forward(q_s, k_s, v_s, self.attention_mask)

        # 4. Concatenate all heads
        # transpose: (batch_size, seq_len_q, n_heads, d_v)
        # .view: (batch_size, seq_len_q, n_heads * d_v)
        concat_heads = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.n_heads * self.d_v
        )

        # 5. Final linear layer back to d_model
        output = self.linear(concat_heads)  # (batch_size, seq_len_q, d_model)

        # 6. Residual connection + LayerNorm
        return self.layerNorm(output + Q), attention_scores

class PoswiseFeedForwardNet(nn.Module):

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.l1 = nn.Linear(d_model, d_ff)
        self.l2 = nn.Linear(d_ff, d_model)

        self.relu = GELU()
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        residual = inputs
        output = self.l1(inputs)
        output = self.relu(output)
        output = self.l2(output)
        return self.layer_norm(output + residual)        


class EncoderLayer(nn.Module):

    def __init__(self, d_model, d_ff, d_k, d_v, n_heads, device):
        super().__init__()
        self.enc_self_attn = MultiHeadAttention(
            d_model=d_model,d_k=d_k,
            d_v=d_v, n_heads=n_heads, device=device)
        self.pos_ffn = PoswiseFeedForwardNet(
            d_model=d_model, d_ff=d_ff)

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(
                Q=enc_inputs, K=enc_inputs,
                V=enc_inputs, attn_mask=enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn



class Encoder(nn.Module):

    def __init__(self, vocab_size, d_model, d_ff, d_k, d_v, n_heads, n_layers, pad_index, device):
        super().__init__()
        self.device = device
        self.pad_index = pad_index
        self.src_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(
            d_model=d_model,
            dropout=0)

        self.layers = []
        for _ in range(n_layers):
            encoder_layer = EncoderLayer(
                d_model=d_model, d_ff=d_ff,
                d_k=d_k, d_v=d_v, n_heads=n_heads,
                device=device)
            self.layers.append(encoder_layer)
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        enc_outputs = self.src_emb(x)
        enc_outputs = self.pos_emb(enc_outputs)
        enc_self_attn_mask = get_attn_pad_mask(x, x, self.pad_index)

        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)

        enc_self_attns = torch.stack(enc_self_attns)
        enc_self_attns = enc_self_attns.permute([1, 0, 2, 3, 4])
        return enc_outputs, enc_self_attns





class DecoderLayer(nn.Module):

    def __init__(self, d_model, d_ff, d_k, d_v, n_heads, device):
        super().__init__()
        self.dec_self_attn = MultiHeadAttention(
            d_model=d_model,d_k=d_k,
            d_v=d_v, n_heads=n_heads, device=device)
        self.dec_enc_attn = MultiHeadAttention(
            d_model=d_model,d_k=d_k,
            d_v=d_v, n_heads=n_heads, device=device)
        self.pos_ffn = PoswiseFeedForwardNet(
            d_model=d_model, d_ff=d_ff)

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
        self.tgt_emb = nn.Embedding(
            vocab_size, d_model)
        self.pos_emb = PositionalEncoding(
            d_model=d_model,
            dropout=0)
        self.layers = []
        for _ in range(n_layers):
            decoder_layer = DecoderLayer(
                d_model=d_model, d_ff=d_ff,
                d_k=d_k, d_v=d_v,
                n_heads=n_heads, device=device)
            self.layers.append(decoder_layer)
        self.layers = nn.ModuleList(self.layers)

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        dec_outputs = self.tgt_emb(dec_inputs)
        dec_outputs = self.pos_emb(dec_outputs)

        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs, self.pad_index)
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)
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
        dec_self_attns = torch.stack(dec_self_attns)
        dec_enc_attns = torch.stack(dec_enc_attns)

        dec_self_attns = dec_self_attns.permute([1, 0, 2, 3, 4])
        dec_enc_attns = dec_enc_attns.permute([1, 0, 2, 3, 4])
        
        return dec_outputs, dec_self_attns, dec_enc_attns




class MaskedDecoderLayer(nn.Module):

    def __init__(self, d_model, d_ff, d_k, d_v, n_heads, device):
        super().__init__()
        self.dec_self_attn = MultiHeadAttention(
            d_model=d_model, d_k=d_k,
            d_v=d_v, n_heads=n_heads, device=device)
        self.pos_ffn = PoswiseFeedForwardNet(
            d_model=d_model, d_ff=d_ff)

    def forward(self, dec_inputs, dec_self_attn_mask):
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn



class MaskedDecoder(nn.Module):

    def __init__(self, vocab_size, d_model, d_ff, d_k,
                 d_v, n_heads, n_layers, pad_index, device):
        super().__init__()
        self.pad_index = pad_index
        self.tgt_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(
            d_model=d_model,
            dropout=0)

        self.layers = []
        for _ in range(n_layers):
            decoder_layer = MaskedDecoderLayer(
                d_model=d_model, d_ff=d_ff,
                d_k=d_k, d_v=d_v, n_heads=n_heads,
                device=device)
            self.layers.append(decoder_layer)
        self.layers = nn.ModuleList(self.layers)

    def forward(self, dec_inputs):
        dec_outputs = self.tgt_emb(dec_inputs)
        dec_outputs = self.pos_emb(dec_outputs)

        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs, self.pad_index)
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)
        dec_self_attns = []
        for layer in self.layers:
            dec_outputs, dec_self_attn = layer(
                dec_inputs=dec_outputs,
                dec_self_attn_mask=dec_self_attn_mask)
            dec_self_attns.append(dec_self_attn)
        dec_self_attns = torch.stack(dec_self_attns)
        dec_self_attns = dec_self_attns.permute([1, 0, 2, 3, 4])
        return dec_outputs, dec_self_attns




class GPTModel(nn.Module):

    def __init__(self, vocab_size, d_model, d_ff,
                 d_k, d_v, n_heads, n_layers, pad_index,
                 device):
        super(GPTModel, self).__init__()
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