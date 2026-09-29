import torch
import torch.nn as nn

from data.tokens import IMAGE
from embedding.clip import CLIPEncoder
from model.language_model import LanguageModel
from model.projector import SpatialMergeProjector


class ModernVLM(nn.Module):
    """Modern VLM (LLaVA-NeXT / Qwen2-VL style).

    Features:
    1. 2x2 Spatial Merging Projector: compresses 4 neighboring image patches into
       1 visual token via pixel unshuffle + MLP, achieving a 4x reduction in sequence
       length while retaining full visual resolution.
    2. Autoregressive Causal Language Model.
    """

    def __init__(
        self,
        height: int = 32,
        width: int = 32,
        patch_size: int = 8,
        in_channels: int = 3,
        vision_width: int = 64,
        vision_heads: int = 4,
        vision_layers: int = 2,
        vocab_size: int = 64,
        max_length: int = 64,
        lang_width: int = 64,
        lang_heads: int = 4,
        lang_layers: int = 2,
        merge_size: int = 2,
        image_token_id: int = IMAGE,
    ):
        super().__init__()
        self.image_token_id = image_token_id
        self.patch_size = patch_size
        self.grid_h = height // patch_size
        self.grid_w = width // patch_size
        self.merge_size = merge_size

        self.vision = CLIPEncoder(
            height=height,
            width=width,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=vision_width,
            num_heads=vision_heads,
            num_layers=vision_layers,
        )

        self.projector = SpatialMergeProjector(
            in_dim=vision_width,
            out_dim=lang_width,
            merge_size=merge_size,
        )

        self.language = LanguageModel(
            vocab_size=vocab_size,
            max_length=max_length,
            embed_dim=lang_width,
            num_heads=lang_heads,
            num_layers=lang_layers,
        )

        self.num_merged_tokens = (self.grid_h // merge_size) * (self.grid_w // merge_size)

    def _vision_tokens(self, images: torch.Tensor) -> torch.Tensor:
        # Get patch tokens, dropping [CLS] token at index 0
        patches = self.vision.forward_tokens(images)[:, 1:, :]
        # Apply 2x2 spatial merge: e.g. (4x4 = 16 patches) -> (2x2 = 4 merged tokens)
        return self.projector(patches, grid_h=self.grid_h, grid_w=self.grid_w)

    def _image_index(self, token_row: torch.Tensor) -> int:
        hits = (token_row == self.image_token_id).nonzero(as_tuple=False)
        if hits.numel() != 1:
            raise ValueError("each sequence needs exactly one image placeholder token")
        return hits.item()

    def merge(self, images: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        text = self.language.embedding.embed_tokens(token_ids)
        vision = self._vision_tokens(images)
        rows = []

        for i in range(token_ids.size(0)):
            idx = self._image_index(token_ids[i])
            rows.append(torch.cat((text[i, :idx], vision[i], text[i, idx + 1:]), dim=0))

        merged = torch.stack(rows, dim=0)
        return self.language.embedding.add_positions(merged)

    def expand_labels(self, token_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        rows = []
        for i in range(token_ids.size(0)):
            idx = self._image_index(token_ids[i])
            ignore = labels.new_full((self.num_merged_tokens,), -100)
            rows.append(torch.cat((labels[i, :idx], ignore, labels[i, idx + 1:])))
        return torch.stack(rows, dim=0)

    def forward(self, images: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        return self.language.forward_embeddings(self.merge(images, token_ids))

    def _append_token(self, embeddings: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        vectors = self.language.embedding.embed_tokens(token_ids.unsqueeze(1)).squeeze(1)
        position = embeddings.size(1)
        pos = self.language.embedding.position(
            torch.tensor([position], device=embeddings.device)
        )
        return torch.cat((embeddings, (vectors + pos).unsqueeze(1)), dim=1)

    @torch.no_grad()
    def generate(self, images: torch.Tensor, token_ids: torch.Tensor, max_new_tokens: int, eos_id: int = None):
        embeddings = self.merge(images, token_ids)
        generated = []

        for _ in range(max_new_tokens):
            if embeddings.size(1) >= self.language.embedding.max_length:
                break
            next_id = self.language.forward_embeddings(embeddings)[:, -1].argmax(dim=-1)
            generated.append(next_id)

            if eos_id is not None and bool((next_id == eos_id).all()):
                break

            embeddings = self._append_token(embeddings, next_id)

        return torch.stack(generated, dim=1)
