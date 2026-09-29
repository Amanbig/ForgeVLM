import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """Image to a sequence of patch vectors (ViT stem).

    How do we feed a 2D image into a 1D Transformer?
    Transformers expect sequences of tokens: (batch, sequence_length, embed_dim).
    An image is a 2D grid of pixels: (batch, channels, height, width).

    The Patch Embedding trick:
      1. Slice the image into non-overlapping square patches of size (P x P).
         For a 32x32 image with patch_size=8, we get (32/8) * (32/8) = 4 * 4 = 16 patches.
      2. Each patch has P * P * C pixels (e.g. 8 * 8 * 3 = 192 values).
      3. A Conv2d with kernel_size=P and stride=P simultaneously extracts each patch
         and projects it to a vector of width `embed_dim`.
    """

    def __init__(self, height: int, width: int, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        assert height % patch_size == 0, f"height ({height}) must be divisible by patch_size ({patch_size})"
        assert width % patch_size == 0, f"width ({width}) must be divisible by patch_size ({patch_size})"

        self.height = height
        self.width = width
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        # Total number of visual patches
        self.num_patches = (height // patch_size) * (width // patch_size)

        # A 2D convolution with stride == kernel_size is mathematically equivalent to
        # extracting non-overlapping patches and multiplying each by a linear weight matrix.
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:

            x: Image tensor of shape (batch, in_channels, height, width)
        Returns:
            Patch tokens of shape (batch, num_patches, embed_dim)
        """
        # Step 1: Convolve over the image
        # (batch, in_channels, height, width) -> (batch, embed_dim, grid_h, grid_w)
        x = self.proj(x)

        # Step 2: Flatten the 2D spatial grid (grid_h, grid_w) into 1D patches
        # (batch, embed_dim, grid_h, grid_w) -> (batch, embed_dim, num_patches)
        x = x.flatten(2)

        # Step 3: Transpose so the sequence length is in dimension 1 (batch, tokens, embed_dim)
        # (batch, embed_dim, num_patches) -> (batch, num_patches, embed_dim)
        x = x.transpose(1, 2)

        return x