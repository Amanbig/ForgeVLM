import torch
import torch.nn as nn

from embedding.clip import CLIPEncoder
from model.language_model import LanguageModel
from model.projector import Projector


class VLM(nn.Module):
    """One causal sequence of words and image patches.

    The text contains a single image placeholder. That one id is replaced by the
    projected patch vectors, then the language model reads the combined stream.
    """

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
        image_token_id,
    ):
        super().__init__()

        self.image_token_id = image_token_id

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

    def _vision_tokens(self, images):
        tokens = self.vision.forward_tokens(images)
        return self.projector(tokens[:, 1:, :])

    def _image_index(self, token_row):
        hits = (token_row == self.image_token_id).nonzero(as_tuple=False)

        if hits.numel() != 1:
            raise ValueError("each sequence needs exactly one image token")

        return hits.item()

    def merge(self, images, token_ids):
        text = self.language.embedding.embed_tokens(token_ids)
        vision = self._vision_tokens(images)
        rows = []

        for i in range(token_ids.size(0)):
            idx = self._image_index(token_ids[i])
            rows.append(torch.cat((text[i, :idx], vision[i], text[i, idx + 1:]), dim=0))

        merged = torch.stack(rows, dim=0)
        return self.language.embedding.add_positions(merged)

    def expand_labels(self, token_ids, labels):
        if labels.shape != token_ids.shape:
            raise ValueError("labels must line up with token_ids, before image patches are inserted")

        n_patches = self.vision.patch_embedding.num_patches
        rows = []

        for i in range(token_ids.size(0)):
            idx = self._image_index(token_ids[i])
            ignore = labels.new_full((n_patches,), -100)
            rows.append(torch.cat((labels[i, :idx], ignore, labels[i, idx + 1:])))

        return torch.stack(rows, dim=0)

    def forward(self, images, token_ids):
        return self.language.forward_embeddings(self.merge(images, token_ids))

    def _append_token(self, embeddings, token_ids):
        vectors = self.language.embedding.embed_tokens(token_ids.unsqueeze(1)).squeeze(1)
        position = embeddings.size(1)
        pos = self.language.embedding.position(
            torch.tensor([position], device=embeddings.device)
        )
        return torch.cat((embeddings, (vectors + pos).unsqueeze(1)), dim=1)

    @torch.no_grad()
    def generate(self, images, token_ids, max_new_tokens, eos_id=None):
        embeddings = self.merge(images, token_ids)
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
