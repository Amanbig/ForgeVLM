import torch
import torch.nn as nn
from .transformer import TransformerBlock
from embedding.patch_embedding import PatchEmbedding


class VisionTransformer(nn.Module):
    """Vision Transformer (ViT) Image Classifier.

    How does ViT work? (Dosovitskiy et al., 2020)
      1. Turn image into a sequence of patch vectors via `PatchEmbedding`.
      2. Prepend a learnable [CLS] (class) token to the patch sequence.
         Because self-attention is permutation-equivariant, the class token can
         attend to all visual patches across layers and summarize the entire image.
      3. Add learned 1D positional embeddings to tell the model where each patch came from.
      4. Pass the sequence through standard pre-norm Transformer blocks (bidirectional attention).
      5. Read out the hidden state of the [CLS] token at position 0, and classify with a Linear head.
    """

    def __init__(
        self,
        height: int,
        width: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        num_classes: int,
    ):
        super().__init__()

        # Step 1: Patch extraction stem
        self.patch_embedding = PatchEmbedding(height, width, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embedding.num_patches

        # Transformer blocks (bidirectional self-attention)
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])

        # Learnable [CLS] token and positional embedding table
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        # Important: Initialize with small random normal values.
        # Zeros would make every position identical, so the transformer has no initial spatial signal!
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embedding, std=0.02)

        # Classification head attached to the [CLS] token representation
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:

            x: Input images of shape (batch, in_channels, height, width)
        Returns:
            Class logits of shape (batch, num_classes)
        """
        batch_size = x.size(0)

        # 1. Image to patches: (batch, num_patches, embed_dim)
        x = self.patch_embedding(x)

        # 2. Prepend [CLS] token: (batch, 1 + num_patches, embed_dim)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # 3. Add positional embeddings (broadcasted across batch)
        x = x + self.pos_embedding

        # 4. Pass through bidirectional Transformer blocks
        for layer in self.layers:
            x = layer(x)

        # 5. Classify using the [CLS] token at index 0: shape (batch, num_classes)
        return self.head(x[:, 0])