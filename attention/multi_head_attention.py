import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.head_dim = embed_dim // num_heads

        self.q = nn.Linear(embed_dim, embed_dim)

        self.k = nn.Linear(embed_dim, embed_dim)

        self.v = nn.Linear(embed_dim, embed_dim)

        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, context=None, attn_mask=None):
        kv = x if context is None else context

        q = self.q(x)
        k = self.k(kv)
        v = self.v(kv)

        b, seq_q, _ = q.shape
        _, seq_k, _ = k.shape

        q = q.view(b, seq_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, seq_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, seq_k, self.num_heads, self.head_dim).transpose(1, 2)

        score = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attn_mask is not None:
            score = score + attn_mask.to(dtype=score.dtype, device=score.device)

        attn = torch.softmax(score, dim=-1)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(b, seq_q, self.embed_dim)
        return self.out(out)