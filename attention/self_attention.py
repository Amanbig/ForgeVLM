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

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        score = q @ k.transpose(-2, -1)

        score = score / math.sqrt(k.size(-1))

        mask = torch.tril(torch.ones(x.size(1), x.size(1)))

        score = score.masked_fill(mask == 0, float('-inf'))

        attn = torch.softmax(score, dim = -1)

        out = attn @ v

        return out