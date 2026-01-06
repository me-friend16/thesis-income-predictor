import torch, torch.nn as nn, math
from einops import rearrange

class ProbAttention(nn.Module):
    def __init__(self, d_model, n_heads, mask_flag=True, factor=5, scale=None, dropout=0.1):
        super().__init__()
        head_dim = d_model // n_heads
        self.scale = scale or 1. / math.sqrt(head_dim)
        self.factor = factor

    def forward(self, q, k, v):
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn   = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, v), attn

class InformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        head_dim = d_model // n_heads
        self.n_heads = n_heads
        self.qkv     = nn.Linear(d_model, d_model * 3)
        self.attn = ProbAttention(d_model, n_heads)
        self.fc      = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, L, d_model]
        B, L, _ = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b l (h d) -> b h l d', h=self.n_heads), qkv)
        out, _ = self.attn(q, k, v)
        out = rearrange(out, 'b h l d -> b l (h d)')
        x = self.norm1(x + self.drop(out))
        ff = self.fc(x)
        return self.norm2(x + self.drop(ff))

class InformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([InformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)])
    def forward(self, x):
        for layer in self.layers: x = layer(x)
        return x

class Informer(nn.Module):
    def __init__(self, enc_in, dec_in, d_model=256, n_heads=8, e_layers=2, d_ff=512, out_len=90):
        super().__init__()
        self.enc_embedding = nn.Linear(enc_in, d_model)
        self.dec_embedding = nn.Linear(dec_in, d_model)
        self.encoder = InformerEncoder(e_layers, d_model, n_heads, d_ff)
        self.projection = nn.Linear(d_model, 1)
        self.out_len = out_len

    def forward(self, x_enc, x_dec):
        # x_enc: [B, L, enc_in]  |  x_dec: [B, out_len, dec_in]
        enc_out = self.enc_embedding(x_enc)          # [B, L, d_model]
        enc_out = self.encoder(enc_out)              # [B, L, d_model]
        pooled  = enc_out.mean(dim=1)                # [B, d_model]
        dec_in  = self.dec_embedding(x_dec)          # [B, out_len, d_model]
        out     = self.projection(pooled).unsqueeze(1).repeat(1, self.out_len, 1)
        return out.squeeze(-1)                       # [B, out_len]
