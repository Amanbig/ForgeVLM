import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.embed_dim = embed_dim

        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)

        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, attn_mask=None):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        score = q @ k.transpose(-2, -1)
        score = score / math.sqrt(k.size(-1))

        if attn_mask is not None:
            score = score + attn_mask.to(dtype=score.dtype, device=score.device)
        else:
            mask = torch.tril(torch.ones(x.size(1), x.size(1), device=x.device))
            score = score.masked_fill(mask == 0, float('-inf'))

        attn = torch.softmax(score, dim=-1)
        out = attn @ v

        return self.out(out)