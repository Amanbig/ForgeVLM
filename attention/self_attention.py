import math
import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    """Single-head scaled dot-product self-attention with causal masking.

    What is Attention?
    Instead of treating tokens independently, attention allows each token in a sequence
    to look at ('attend to') all preceding tokens and gather relevant context.

    The three roles:
      - Query (Q): What this token is currently searching for.
      - Key   (K): What this token contains (the 'index' or 'tag').
      - Value (V): The actual content or information payload this token holds.

    Attention Formula:
      Attention(Q, K, V) = softmax( (Q @ K^T) / sqrt(d_k) ) @ V
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

        # Linear projections from token embedding to Q, K, V vectors
        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)

        # Final output projection to mix the attended representations
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        """Args:

            x: Input sequence of shape (batch_size, seq_len, embed_dim)
            attn_mask: Optional attention mask (0 = keep, -inf = mask out)
        Returns:
            Output sequence of shape (batch_size, seq_len, embed_dim)
        """
        # Step 1: Project inputs to Query, Key, and Value
        # Shapes: (batch_size, seq_len, embed_dim)
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        # Step 2: Compute raw compatibility scores between every query and key
        # (batch_size, seq_len, embed_dim) @ (batch_size, embed_dim, seq_len)
        # -> shape: (batch_size, seq_len, seq_len)
        # Entry (i, j) is the dot product of token i's query and token j's key.
        score = q @ k.transpose(-2, -1)

        # Step 3: Scale by sqrt(d_k) to prevent dot products from growing too large
        # Large dot products would push the softmax into tiny gradients (vanishing gradients).
        score = score / math.sqrt(k.size(-1))

        # Step 4: Apply attention mask
        if attn_mask is not None:
            # Custom mask (e.g. prefix mask or bidirectional mask)
            score = score + attn_mask.to(dtype=score.dtype, device=score.device)
        else:
            # Default: Causal mask (lower triangular matrix)
            # Token i can only attend to past tokens 0..i, not future tokens i+1..N.
            mask = torch.tril(torch.ones(x.size(1), x.size(1), device=x.device))
            score = score.masked_fill(mask == 0, float("-inf"))

        # Step 5: Convert scores to probabilities summing to 1 across the row
        # shape: (batch_size, seq_len, seq_len)
        attn = torch.softmax(score, dim=-1)

        # Step 6: Compute weighted sum of values
        # (batch_size, seq_len, seq_len) @ (batch_size, seq_len, embed_dim)
        # -> shape: (batch_size, seq_len, embed_dim)
        out = attn @ v

        # Step 7: Final linear projection
        return self.out(out)