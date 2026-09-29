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

    def forward(self, x, attn_mask=None):

        # Attention
        h = self.norm1(x)

        if attn_mask is not None:
            attn_mask = attn_mask.to(dtype=h.dtype, device=h.device)

        attn_out, _ = self.attn(
            h, h, h,
            attn_mask=attn_mask,
        )

        x = x + attn_out

        # MLP
        h = self.norm2(x)

        x = x + self.mlp(h)

        return x


def causal_mask(seq_len, device):
    # 0 keeps a token, -inf hides it. The diagonal stays visible.
    future = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    return future.masked_fill(future == 1, float("-inf"))


def prefix_mask(n_prefix, n_text, device):
    # Prefix tokens see each other. Text tokens see the whole prefix and earlier text.
    total = n_prefix + n_text
    allowed = torch.zeros(total, total, device=device)
    allowed[:n_prefix, :n_prefix] = 1

    if n_text:
        allowed[n_prefix:, :n_prefix] = 1
        allowed[n_prefix:, n_prefix:] = torch.tril(torch.ones(n_text, n_text, device=device))

    mask = torch.zeros(total, total, device=device)
    return mask.masked_fill(allowed == 0, float("-inf"))