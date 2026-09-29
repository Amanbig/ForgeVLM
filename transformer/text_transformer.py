import torch
import torch.nn as nn

from embedding.token_embedding import TokenEmbedding
from transformer.transformer import TransformerBlock, causal_mask


class TextTransformer(nn.Module):
    """Text encoder. Reads the end-of-text position, one vector per caption."""

    def __init__(self, vocab_size, max_length, embed_dim, num_heads, num_layers):
        super().__init__()

        self.embedding = TokenEmbedding(vocab_size, embed_dim, max_length)

        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, token_ids):
        x = self.embedding(token_ids)

        mask = causal_mask(x.size(1), x.device)

        for layer in self.layers:
            x = layer(x, attn_mask=mask)

        x = self.norm(x)

        # Arrange the vocabulary so the end-of-text id is the largest id.
        # argmax then finds that token, which has seen the whole caption.
        eot = token_ids.argmax(dim=-1)
        return x[torch.arange(x.size(0), device=x.device), eot]
