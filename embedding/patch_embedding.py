import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, height, width, patch_size, in_channels, embed_dim):
        super().__init__()

        self.height = height
        self.width = width
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.num_patches = (height // patch_size) * (width // patch_size)

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)

        x = x.flatten(2)

        x = x.transpose(1, 2)

        return x