import torch
import torch.nn as nn

from embedding.patch_embedding import PatchEmbedding
from transformer.transformer import TransformerBlock


class CLIPEncoder(nn.Module):
    """Image encoder. One vector per image, read from the class token."""

    def __init__(self, height, width, patch_size, in_channels, embed_dim, num_heads, num_layers):
        super().__init__()

        self.patch_embedding = PatchEmbedding(height, width, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embedding.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        # Zeros would make every position identical, so the block would have no location signal.
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embedding, std=0.02)

        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)

    def forward_tokens(self, x):
        if x.shape[-2] != self.patch_embedding.height or x.shape[-1] != self.patch_embedding.width:
            raise ValueError(
                f"expected image {(self.patch_embedding.height, self.patch_embedding.width)}, "
                f"got {tuple(x.shape[-2:])}"
            )

        batch_size = x.size(0)

        x = self.patch_embedding(x)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding

        for layer in self.layers:
            x = layer(x)

        return self.norm(x)

    def forward(self, x):
        return self.forward_tokens(x)[:, 0]
