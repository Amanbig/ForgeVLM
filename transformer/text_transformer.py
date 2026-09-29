import torch
import torch.nn as nn

from embedding.token_embedding import TokenEmbedding
from transformer.transformer import TransformerBlock, causal_mask


class TextTransformer(nn.Module):
    """CLIP Causal Text Encoder.

    How does the text encoder summarize a caption into one vector?
      1. Turn token IDs into vectors and add positional embeddings via `TokenEmbedding`.
      2. Pass through causal Transformer blocks (each word can only see words before it).
      3. Locate the End-Of-Text (<eos>) token.
         Because of the causal mask, the <eos> token has attended over the ENTIRE caption.
         Its hidden state serves as the global text representation, mirroring the [CLS]
         token in the vision tower!
    """

    def __init__(self, vocab_size: int, max_length: int, embed_dim: int, num_heads: int, num_layers: int):
        super().__init__()

        # Embeds discrete token IDs and adds position vectors
        self.embedding = TokenEmbedding(vocab_size, embed_dim, max_length)

        # Causal transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])

        # Final layer normalization
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Args:

            token_ids: Caption token IDs of shape (batch, seq_len)
        Returns:
            Sentence summary vectors of shape (batch, embed_dim)
        """
        # Step 1: Token embedding + position embedding: (batch, seq_len, embed_dim)
        x = self.embedding(token_ids)

        # Step 2: Create causal mask on the correct device
        mask = causal_mask(x.size(1), x.device)

        # Step 3: Pass through causal Transformer layers
        for layer in self.layers:
            x = layer(x, attn_mask=mask)

        # Step 4: Final LayerNorm
        x = self.norm(x)

        # Step 5: Read the vector at the End-Of-Text (<eos>) token position.
        # By convention, <eos> has the largest ID in the vocabulary.
        # argmax along the sequence dimension finds its exact index in each row.
        eot = token_ids.argmax(dim=-1)  # shape: (batch,)

        # Extract the hidden states at those indices: shape (batch, embed_dim)
        return x[torch.arange(x.size(0), device=x.device), eot]
