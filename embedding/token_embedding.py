import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """Token ids to vectors, plus a learned position for each index."""

    def __init__(self, vocab_size, embed_dim, max_length):
        super().__init__()

        self.max_length = max_length

        self.token = nn.Embedding(vocab_size, embed_dim)
        self.position = nn.Embedding(max_length, embed_dim)

    def forward(self, token_ids):
        length = token_ids.size(1)

        if length > self.max_length:
            raise ValueError(f"sequence length {length} is longer than max_length {self.max_length}")

        positions = torch.arange(length, device=token_ids.device)
        return self.token(token_ids) + self.position(positions)
