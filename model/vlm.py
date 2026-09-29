import torch
import torch.nn as nn

from embedding.clip import CLIPEncoder
from model.language_model import LanguageModel
from model.projector import Projector


class VLM(nn.Module):
    """Vision-Language Model (LLaVA-1.0 style).

    How does a VLM understand an image?
      The model treats image patches as visual 'words'.
      1. Slices the image into patches and extracts features via a vision encoder (`CLIPEncoder`).
      2. Projects the vision features into the language model's dimension via an MLP (`Projector`).
      3. In the user's text prompt, a single special placeholder token `<image>` represents the image.
      4. `merge(...)` finds where `<image>` is located and splices the projected visual patch vectors
         directly into the sequence of word vectors!
      5. The causal language model reads the combined stream: [words... image_patches... question...]
         and predicts the assistant's answer word by word.
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
        max_length: int,
        lang_width: int,
        lang_heads: int,
        lang_layers: int,
        image_token_id: int,
    ):
        super().__init__()
        self.image_token_id = image_token_id

        # Vision Encoder (ViT stem)
        self.vision = CLIPEncoder(
            height,
            width,
            patch_size,
            in_channels,
            vision_width,
            vision_heads,
            vision_layers,
        )

        # Multimodal Projector: maps from vision_width to lang_width
        self.projector = Projector(vision_width, lang_width)

        # Causal Language Model
        self.language = LanguageModel(
            vocab_size,
            max_length,
            lang_width,
            lang_heads,
            lang_layers,
        )

    def _vision_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """Extracts patch tokens, drops [CLS] token at index 0, and projects to language width."""
        tokens = self.vision.forward_tokens(images)
        # Drop index 0 ([CLS]), keep indices 1..N (patches)
        # (batch, num_patches, vision_width) -> (batch, num_patches, lang_width)
        return self.projector(tokens[:, 1:, :])

    def _image_index(self, token_row: torch.Tensor) -> int:
        """Finds the index of the <image> placeholder token in a sequence."""
        hits = (token_row == self.image_token_id).nonzero(as_tuple=False)
        if hits.numel() != 1:
            raise ValueError("each sequence needs exactly one image placeholder token")
        return hits.item()

    def merge(self, images: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        """Splices projected image patches into the text embedding stream.

        Visual Diagram:
          Before merge: [BOS, USER, <image>, "What", "color?", ASSISTANT]
          Splicing:     [BOS, USER] + [patch_1, ..., patch_16] + ["What", "color?", ASSISTANT]
          After merge:  Full combined embedding sequence with position vectors added.
        """
        # Step 1: Embed text tokens (batch, seq_len, lang_width)
        text = self.language.embedding.embed_tokens(token_ids)

        # Step 2: Extract and project visual patches (batch, num_patches, lang_width)
        vision = self._vision_tokens(images)
        rows = []

        # Step 3: Replace the <image> placeholder in each batch element
        for i in range(token_ids.size(0)):
            idx = self._image_index(token_ids[i])
            # Concatenate: text before image + all image patches + text after image
            rows.append(torch.cat((text[i, :idx], vision[i], text[i, idx + 1:]), dim=0))

        merged = torch.stack(rows, dim=0)

        # Step 4: Add learned positional embeddings across the combined sequence
        return self.language.embedding.add_positions(merged)

    def expand_labels(self, token_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Expands target label rows to line up with the merged image patches.

        Why do we need this?
        The original `labels` tensor only matches the text length. When we insert
        `n_patches` into the sequence, the model outputs logits for those patch positions too.
        We insert `-100` (ignore index) at all patch positions so the loss ONLY trains
        on predicting text, not predicting image patches!
        """
        if labels.shape != token_ids.shape:
            raise ValueError("labels must line up with token_ids, before image patches are inserted")

        n_patches = self.vision.patch_embedding.num_patches
        rows = []

        for i in range(token_ids.size(0)):
            idx = self._image_index(token_ids[i])
            ignore = labels.new_full((n_patches,), -100)
            rows.append(torch.cat((labels[i, :idx], ignore, labels[i, idx + 1:])))

        return torch.stack(rows, dim=0)

    def forward(self, images: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        """Computes vocabulary logits over the merged multimodal sequence."""
        return self.language.forward_embeddings(self.merge(images, token_ids))

    def _append_token(self, embeddings: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        """Appends newly generated token embeddings to the running sequence during generation."""
        vectors = self.language.embedding.embed_tokens(token_ids.unsqueeze(1)).squeeze(1)
        position = embeddings.size(1)
        pos = self.language.embedding.position(
            torch.tensor([position], device=embeddings.device)
        )
        return torch.cat((embeddings, (vectors + pos).unsqueeze(1)), dim=1)

    @torch.no_grad()
    def generate(self, images: torch.Tensor, token_ids: torch.Tensor, max_new_tokens: int, eos_id: int = None):
        """Autoregressively generates text tokens one word at a time given an image prompt.

        Loop:
          1. Run forward pass on current sequence embeddings.
          2. Take logits at the LAST position: `logits[:, -1]`.
          3. Pick the most likely token ID via argmax (greedy decoding).
          4. Append new token to embeddings and repeat until max_new_tokens or <eos>.
        """
        embeddings = self.merge(images, token_ids)
        generated = []

        for _ in range(max_new_tokens):
            if embeddings.size(1) >= self.language.embedding.max_length:
                raise ValueError("generation ran past max_length")

            # Predict next token from the last position
            next_id = self.language.forward_embeddings(embeddings)[:, -1].argmax(dim=-1)
            generated.append(next_id)

            # Stop early if all batch elements produced <eos>
            if eos_id is not None and bool((next_id == eos_id).all()):
                break

            # Append the predicted token to the embeddings stream
            embeddings = self._append_token(embeddings, next_id)

        return torch.stack(generated, dim=1)
