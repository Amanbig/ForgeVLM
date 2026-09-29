import torch
import torch.nn as nn

from embedding.clip import CLIPEncoder
from model.language_model import LanguageModel
from model.projector import Projector
from transformer.transformer import prefix_mask


class VLM(nn.Module):
    """Vision tokens, projected, then prefixed onto a causal language model."""

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
        max_length,
        lang_width,
        lang_heads,
        lang_layers,
    ):
        super().__init__()

        self.vision = CLIPEncoder(
            height,
            width,
            patch_size,
            in_channels,
            vision_width,
            vision_heads,
            vision_layers,
        )

        self.projector = Projector(vision_width, lang_width)

        self.language = LanguageModel(
            vocab_size,
            max_length,
            lang_width,
            lang_heads,
            lang_layers,
        )

    def forward(self, images, token_ids):
        vision_tokens = self.vision.forward_tokens(images)[:, 1:, :]
        vision_tokens = self.projector(vision_tokens)
        text_tokens = self.language.embed(token_ids)

        embeddings = torch.cat([vision_tokens, text_tokens], dim=1)
        mask = prefix_mask(vision_tokens.size(1), text_tokens.size(1), embeddings.device)
        return self.language.forward_embeddings(embeddings, attn_mask=mask)
