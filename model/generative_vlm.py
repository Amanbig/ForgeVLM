import torch
import torch.nn as nn

from data.tokens import BOI, EOI, IMAGE
from model.language_model import LanguageModel
from model.projector import Projector
from model.vqvae import VQVAE


class GenerativeVLM(nn.Module):
    """Unified Multimodal Generative Model (Chameleon / Show-o style).

    Shares a single autoregressive transformer for both understanding (image -> text)
    and generation (text -> image).
    Visual tokens are discrete codebook indices produced and decoded by a VQ-VAE.
    """

    def __init__(
        self,
        text_vocab_size: int = 64,
        num_visual_tokens: int = 64,
        max_length: int = 96,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        vqvae: VQVAE = None,
    ):
        super().__init__()
        self.text_vocab_size = text_vocab_size
        self.num_visual_tokens = num_visual_tokens
        # Total vocab: [0 .. text_vocab-1] for text, [text_vocab .. text_vocab + visual_tokens - 1] for visual
        self.total_vocab = text_vocab_size + num_visual_tokens

        self.language = LanguageModel(
            vocab_size=self.total_vocab,
            max_length=max_length,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
        )

        self.vqvae = vqvae if vqvae is not None else VQVAE(
            in_channels=3,
            hidden_dim=32,
            latent_dim=16,
            num_embeddings=num_visual_tokens,
        )

    def image_to_visual_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """Converts RGB images into vocabulary-offset discrete visual tokens."""
        indices = self.vqvae.encode_to_indices(images)  # (B, H_grid, W_grid)
        flat_indices = indices.flatten(1)  # (B, num_patches)
        # Shift indices into the visual portion of the unified vocabulary
        return flat_indices + self.text_vocab_size

    def visual_tokens_to_image(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        """Decodes vocabulary-offset discrete visual tokens back into RGB images."""
        raw_indices = visual_tokens - self.text_vocab_size
        return self.vqvae.decode_from_indices(raw_indices)

    def forward(self, token_ids: torch.Tensor):
        """Standard next-token prediction across text and visual tokens."""
        return self.language(token_ids)

    @torch.no_grad()
    def generate_image_tokens(self, prompt_token_ids: torch.Tensor, num_visual_tokens: int):
        """Given a text prompt ending with BOI, generates the visual tokens autoregressively."""
        curr = prompt_token_ids
        generated = []

        for _ in range(num_visual_tokens):
            if curr.size(1) >= self.language.embedding.max_length:
                break
            logits = self.language(curr)[:, -1]
            # Restrict sampling to the visual token partition
            visual_logits = logits[:, self.text_vocab_size : self.text_vocab_size + self.num_visual_tokens]
            next_visual_idx = visual_logits.argmax(dim=-1) + self.text_vocab_size

            generated.append(next_visual_idx)
            curr = torch.cat((curr, next_visual_idx.unsqueeze(1)), dim=1)

        return torch.stack(generated, dim=1)

    @torch.no_grad()
    def generate_image(self, prompt_token_ids: torch.Tensor, num_visual_tokens: int):
        """Autoregressively generates visual tokens and decodes them into RGB images."""
        gen_tokens = self.generate_image_tokens(prompt_token_ids, num_visual_tokens)
        images = self.visual_tokens_to_image(gen_tokens)
        return images, gen_tokens
