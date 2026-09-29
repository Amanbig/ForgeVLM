import torch
import torch.nn as nn

class TransformerBlock(nn.Module):

    def __init__(self, embed_dim, num_heads):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )

    def forward(self, x):

        # Attention
        h = self.norm1(x)

        attn_out, _ = self.attn(
            h, h, h
        )

        x = x + attn_out

        # MLP
        h = self.norm2(x)

        x = x + self.mlp(h)

        return x