import torch.nn as nn

from embedding.token_embedding import TokenEmbedding
from transformer.transformer import TransformerBlock, causal_mask


class LanguageModel(nn.Module):
    """Small causal language model. The token table and the output layer share weights."""

    def __init__(self, vocab_size, max_length, embed_dim, num_heads, num_layers):
        super().__init__()

        self.embedding = TokenEmbedding(vocab_size, embed_dim, max_length)

        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.head.weight = self.embedding.token.weight

    def embed(self, token_ids):
        return self.embedding(token_ids)

    def forward_embeddings(self, embeddings, attn_mask=None):
        if attn_mask is None:
            attn_mask = causal_mask(embeddings.size(1), embeddings.device)

        x = embeddings

        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)

        return self.head(self.norm(x))

    def forward(self, token_ids):
        return self.forward_embeddings(self.embed(token_ids))
