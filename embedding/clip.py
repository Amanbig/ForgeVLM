import torch
import torch.nn as nn

from embedding.patch_embedding import PatchEmbedding
from transformer.transformer import TransformerBlock


class CLIPEncoder(nn.Module):
    """CLIP Vision Encoder (Radford et al., 2021).

    Produces visual representations at two granularities:
      1. `forward_tokens(images)`: Returns ALL patch representations
         Shape: (batch, 1 + num_patches, embed_dim).
         Used by VLMs (like LLaVA) where each patch acts as a visual 'word'.
      2. `forward(images)`: Returns ONLY the [CLS] class token
         Shape: (batch, embed_dim).
         Used by CLIP to compute a single global cosine similarity vector for the whole image.
    """

    def __init__(self, height: int, width: int, patch_size: int, in_channels: int, embed_dim: int, num_heads: int, num_layers: int):
        super().__init__()

        # Slices 2D images into 1D patch tokens
        self.patch_embedding = PatchEmbedding(height, width, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embedding.num_patches

        # Learnable [CLS] summary token and positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        # Initialize with standard deviation 0.02 so patches have initial location signals
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embedding, std=0.02)

        # Stack of Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])

        # Final layer normalization
        self.norm = nn.LayerNorm(embed_dim)

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Processes images and returns all visual tokens (CLS + patches).

        Args:
            x: Images of shape (batch, in_channels, height, width)
        Returns:
            (batch, 1 + num_patches, embed_dim)
        """
        if x.shape[-2] != self.patch_embedding.height or x.shape[-1] != self.patch_embedding.width:
            raise ValueError(
                f"expected image {(self.patch_embedding.height, self.patch_embedding.width)}, "
                f"got {tuple(x.shape[-2:])}"
            )

        batch_size = x.size(0)

        # 1. Image to patch tokens: (batch, num_patches, embed_dim)
        x = self.patch_embedding(x)

        # 2. Prepend [CLS] token: (batch, 1 + num_patches, embed_dim)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # 3. Add learned position vectors
        x = x + self.pos_embedding

        # 4. Bidirectional self-attention blocks
        for layer in self.layers:
            x = layer(x)

        # 5. Final LayerNorm across all tokens
        return self.norm(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the global image summary vector (the [CLS] token at index 0).

        Shape: (batch, embed_dim)
        """
        return self.forward_tokens(x)[:, 0]
