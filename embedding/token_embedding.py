import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """Token ids to vectors, plus a learned position for each index."""

    def __init__(self, vocab_size, embed_dim, max_length):
        super().__init__()

        self.max_length = max_length

        self.token = nn.Embedding(vocab_size, embed_dim)
        self.position = nn.Embedding(max_length, embed_dim)

    def embed_tokens(self, token_ids):
        return self.token(token_ids)

    def add_positions(self, embeddings):
        length = embeddings.size(1)

        if length > self.max_length:
            raise ValueError(f"sequence length {length} is longer than max_length {self.max_length}")

        positions = torch.arange(length, device=embeddings.device)
        return embeddings + self.position(positions)

    def forward(self, token_ids):
        return self.add_positions(self.embed_tokens(token_ids))
