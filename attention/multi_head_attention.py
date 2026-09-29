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

    def forward(self, x):

        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        q = q.view(x.size(0), x.size(1), self.num_heads, self.head_dim).transpose(1 ,2 )
        k = k.view(x.size(0), x.size(1), self.num_heads, self.head_dim).transpose(1 ,2 )
        v = v.view(x.size(0), x.size(1), self.num_heads, self.head_dim).transpose(1 ,2 )

        score = q @ k.transpose(-2 , -1)

        mask = torch.tril(torch.ones(x.size(1), x.size(1)))

        score = score / math.sqrt(self.head_dim)

        score = score.masked_fill(mask == 0, float('-inf'))

        attn = torch.softmax(score, dim = -1)

        out = attn @ v

        out = out.transpose(1 ,2)

        out = out.contiguous().view(x.size(0), x.size(1), self.embed_dim)

        out = self.out(out)

        return out