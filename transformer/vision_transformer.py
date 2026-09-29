import torch
import torch.nn as nn
from .transformer import TransformerBlock
from embedding.patch_embedding import PatchEmbedding

class VisionTransformer(nn.Module):
    def __init__(self, height, width, patch_size, in_channels, embed_dim, num_heads, num_layers, num_classes):
        super().__init__()

        self.patch_embedding = PatchEmbedding(height, width, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embedding.num_patches

        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        batch_size = x.size(0)

        x = self.patch_embedding(x)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embedding

        for layer in self.layers:
            x = layer(x)

        return self.head(x[:, 0])