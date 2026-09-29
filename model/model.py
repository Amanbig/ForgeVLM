import math

import torch
import torch.nn as nn

from embedding.clip import CLIPEncoder
from transformer.text_transformer import TextTransformer


def l2_normalize(x):
    return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-6)


class CLIP(nn.Module):
    """Image tower and text tower, scored by scaled cosine similarity."""

    def __init__(
        self,
        height,
        width,
        patch_size,
        in_channels,
        vision_width,
        vision_heads,
        vision_layers,
        vocab_size,
        context_length,
        text_width,
        text_heads,
        text_layers,
        projection_dim,
    ):
        super().__init__()

        self.image_encoder = CLIPEncoder(
            height,
            width,
            patch_size,
            in_channels,
            vision_width,
            vision_heads,
            vision_layers,
        )

        self.text_encoder = TextTransformer(
            vocab_size,
            context_length,
            text_width,
            text_heads,
            text_layers,
        )

        self.image_projection = nn.Linear(vision_width, projection_dim, bias=False)
        self.text_projection = nn.Linear(text_width, projection_dim, bias=False)

        # Stored in log space. exp(logit_scale) is the temperature multiplier.
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def encode_image(self, images):
        features = self.image_encoder(images)
        return l2_normalize(self.image_projection(features))

    def encode_text(self, token_ids):
        features = self.text_encoder(token_ids)
        return l2_normalize(self.text_projection(features))

    def forward(self, images, token_ids):
        image_features = self.encode_image(images)
        text_features = self.encode_text(token_ids)

        scale = self.logit_scale.exp().clamp(max=100)
        logits_per_image = scale * image_features @ text_features.transpose(-2, -1)
        logits_per_text = logits_per_image.transpose(-2, -1)
        return logits_per_image, logits_per_text
