import math
import torch
import torch.nn as nn

from embedding.clip import CLIPEncoder
from transformer.text_transformer import TextTransformer


def l2_normalize(x: torch.Tensor) -> torch.Tensor:
    """L2 normalizes vectors along the last dimension so their length is exactly 1.0.

    When vectors u and v have unit length (||u||=1, ||v||=1), their dot product
    u @ v equals their cosine similarity, strictly bounded in [-1.0, 1.0].
    """
    return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-6)


class CLIP(nn.Module):
    """Contrastive Language-Image Pre-training (CLIP).

    CLIP aligns images and text into a shared multi-dimensional embedding space:
      - If image i matches text i, their embedding vectors should point in the SAME direction (dot product ~ +1).
      - If image i does not match text j, their vectors should be orthogonal or opposite (dot product ~ 0 or -1).

    How it works:
      1. Vision tower encodes images: (B, C, H, W) -> (B, projection_dim).
      2. Text tower encodes captions: (B, seq_len) -> (B, projection_dim).
      3. Compute (B, B) pairwise cosine similarity matrix.
      4. Scale by a learnable temperature multiplier: exp(logit_scale).
    """

    def __init__(
        self,
        height: int,
        width: int,
        patch_size: int,
        in_channels: int,
        vision_width: int,
        vision_heads: int,
        vision_layers: int,
        vocab_size: int,
        context_length: int,
        text_width: int,
        text_heads: int,
        text_layers: int,
        projection_dim: int,
    ):
        super().__init__()

        # Image tower (ViT stem)
        self.image_encoder = CLIPEncoder(
            height,
            width,
            patch_size,
            in_channels,
            vision_width,
            vision_heads,
            vision_layers,
        )

        # Text tower (Causal Transformer)
        self.text_encoder = TextTransformer(
            vocab_size,
            context_length,
            text_width,
            text_heads,
            text_layers,
        )

        # Projections into the shared embedding dimension (bias-free)
        self.image_projection = nn.Linear(vision_width, projection_dim, bias=False)
        self.text_projection = nn.Linear(text_width, projection_dim, bias=False)

        # Learnable temperature parameter, stored in log space to guarantee positive scale.
        # Initialized to ln(1 / 0.07) ≈ 2.659, so exp(logit_scale) starts near 14.3.
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encodes images into L2-normalized vectors of shape (batch, projection_dim)."""
        features = self.image_encoder(images)
        return l2_normalize(self.image_projection(features))

    def encode_text(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Encodes captions into L2-normalized vectors of shape (batch, projection_dim)."""
        features = self.text_encoder(token_ids)
        return l2_normalize(self.text_projection(features))

    def forward(self, images: torch.Tensor, token_ids: torch.Tensor):
        """Computes pairwise similarity matrices:

        logits_per_image[i, j]: score of image i with caption j.
        logits_per_text[j, i]:  score of caption j with image i.
        """
        # Step 1: Encode and normalize both modalities: (batch, projection_dim)
        image_features = self.encode_image(images)
        text_features = self.encode_text(token_ids)

        # Step 2: Compute scaled cosine similarities
        # Clamp temperature at 100 to prevent training instability/overflow
        scale = self.logit_scale.exp().clamp(max=100)

        # (batch, projection_dim) @ (projection_dim, batch) -> (batch, batch)
        logits_per_image = scale * (image_features @ text_features.transpose(-2, -1))
        logits_per_text = logits_per_image.transpose(-2, -1)

        return logits_per_image, logits_per_text
