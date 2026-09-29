import torch
import torch.nn as nn

from embedding.token_embedding import TokenEmbedding
from transformer.transformer import TransformerBlock, causal_mask


class LanguageModel(nn.Module):
    """Small causal autoregressive Language Model (Decoder-only Transformer).

    Key features:
      1. Causal Masking: Position t can only attend to positions <= t.
      2. Weight Tying: The output linear head shares its weights with the input token
         embedding table (`self.head.weight = self.embedding.token.weight`).
         This halves vocabulary parameters and improves language generalization.
      3. Embedding Injection: Exposes `forward_embeddings` so multimodal models
         (like VLMs) can inject vision or video embeddings directly into the stream!
    """

    def __init__(self, vocab_size: int, max_length: int, embed_dim: int, num_heads: int, num_layers: int):
        super().__init__()

        # Input token embedding + positional table
        self.embedding = TokenEmbedding(vocab_size, embed_dim, max_length)

        # Stack of causal Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])

        # Final LayerNorm before prediction head
        self.norm = nn.LayerNorm(embed_dim)

        # Output projection to vocabulary logits
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        # Weight tying: reading a word and scoring a word use the exact same matrix
        self.head.weight = self.embedding.token.weight

    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Embeds integer token IDs to continuous vectors."""
        return self.embedding(token_ids)

    def forward_embeddings(self, embeddings: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        """Processes a sequence of continuous embeddings through the causal Transformer layers.

        Used by VLMs after inserting projected visual tokens into the prompt sequence.
        Args:
            embeddings: (batch, seq_len, embed_dim)
            attn_mask: Optional attention mask (defaults to causal mask)
        Returns:
            Vocabulary logits of shape (batch, seq_len, vocab_size)
        """
        if attn_mask is None:
            attn_mask = causal_mask(embeddings.size(1), embeddings.device)

        x = embeddings
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)

        # Project final normalized representations to vocabulary logits
        return self.head(self.norm(x))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Standard language-only forward pass from token IDs to logits."""
        return self.forward_embeddings(self.embed(token_ids))
