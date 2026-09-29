import torch
import torch.nn as nn

from data.tokens import VIDEO
from embedding.video_embedding import VideoCLIPEncoder
from model.language_model import LanguageModel
from model.projector import Projector


class VideoVLM(nn.Module):
    """Video Vision-Language Model.

    Reads a video of shape (batch, channels, time, height, width), converts it into
    spatio-temporal tubelet tokens, projects them into language space, replaces the
    `<video>` token placeholder, and autoregressively reasons over the video.
    """

    def __init__(
        self,
        num_frames: int = 4,
        height: int = 32,
        width: int = 32,
        tubelet_time: int = 2,
        patch_size: int = 8,
        in_channels: int = 3,
        vision_width: int = 64,
        vision_heads: int = 4,
        vision_layers: int = 2,
        vocab_size: int = 64,
        max_length: int = 96,
        lang_width: int = 64,
        lang_heads: int = 4,
        lang_layers: int = 2,
        video_token_id: int = VIDEO,
    ):
        super().__init__()
        self.video_token_id = video_token_id

        self.vision = VideoCLIPEncoder(
            num_frames=num_frames,
            height=height,
            width=width,
            tubelet_time=tubelet_time,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=vision_width,
            num_heads=vision_heads,
            num_layers=vision_layers,
        )

        self.projector = Projector(vision_width, lang_width)

        self.language = LanguageModel(
            vocab_size=vocab_size,
            max_length=max_length,
            embed_dim=lang_width,
            num_heads=lang_heads,
            num_layers=lang_layers,
        )

    def _video_tokens(self, videos: torch.Tensor) -> torch.Tensor:
        # videos: (batch, channels, time, height, width)
        tokens = self.vision.forward_tokens(videos)
        # Drop the [CLS] token (position 0), keep the tubelets
        return self.projector(tokens[:, 1:, :])

    def _video_index(self, token_row: torch.Tensor) -> int:
        hits = (token_row == self.video_token_id).nonzero(as_tuple=False)
        if hits.numel() != 1:
            raise ValueError("each sequence needs exactly one video placeholder token")
        return hits.item()

    def merge(self, videos: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        text = self.language.embedding.embed_tokens(token_ids)
        video_tokens = self._video_tokens(videos)
        rows = []

        for i in range(token_ids.size(0)):
            idx = self._video_index(token_ids[i])
            rows.append(torch.cat((text[i, :idx], video_tokens[i], text[i, idx + 1:]), dim=0))

        merged = torch.stack(rows, dim=0)
        return self.language.embedding.add_positions(merged)

    def expand_labels(self, token_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if labels.shape != token_ids.shape:
            raise ValueError("labels must line up with token_ids, before video tubelets are inserted")

        n_tubelets = self.vision.patch_embedding.num_tubelets
        rows = []

        for i in range(token_ids.size(0)):
            idx = self._video_index(token_ids[i])
            ignore = labels.new_full((n_tubelets,), -100)
            rows.append(torch.cat((labels[i, :idx], ignore, labels[i, idx + 1:])))

        return torch.stack(rows, dim=0)

    def forward(self, videos: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        return self.language.forward_embeddings(self.merge(videos, token_ids))

    def _append_token(self, embeddings: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        vectors = self.language.embedding.embed_tokens(token_ids.unsqueeze(1)).squeeze(1)
        position = embeddings.size(1)
        pos = self.language.embedding.position(
            torch.tensor([position], device=embeddings.device)
        )
        return torch.cat((embeddings, (vectors + pos).unsqueeze(1)), dim=1)

    @torch.no_grad()
    def generate(self, videos: torch.Tensor, token_ids: torch.Tensor, max_new_tokens: int, eos_id: int = None):
        embeddings = self.merge(videos, token_ids)
        generated = []

        for _ in range(max_new_tokens):
            if embeddings.size(1) >= self.language.embedding.max_length:
                raise ValueError("generation ran past max_length")

            next_id = self.language.forward_embeddings(embeddings)[:, -1].argmax(dim=-1)
            generated.append(next_id)

            if eos_id is not None and bool((next_id == eos_id).all()):
                break

            embeddings = self._append_token(embeddings, next_id)

        return torch.stack(generated, dim=1)
