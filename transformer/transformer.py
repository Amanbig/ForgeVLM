import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    """Pre-Norm Transformer Block.

    Architecture (Pre-LN):
      Input x
         |------> LayerNorm -> Multi-Head Attention -> + (Residual)
         |                                              |
         |----------------------------------------------|
         |
         |------> LayerNorm -> FeedForward (MLP) ----> + (Residual)
         |                                              |
         |----------------------------------------------|
         |
      Output x

    Why Pre-LN instead of original Post-LN?
    Original Transformers (Vaswani et al., 2017) applied LayerNorm AFTER the residual connection.
    Modern models (GPT-3, LLaMA, ViT) apply LayerNorm BEFORE attention and MLP. Pre-LN provides
    an unobstructed gradient highway along the residual stream, enabling stable training without
    warmup tricks.
    """

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()

        # Normalization before attention
        self.norm1 = nn.LayerNorm(embed_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            batch_first=True,
        )

        # Normalization before MLP
        self.norm2 = nn.LayerNorm(embed_dim)

        # Multi-Layer Perceptron (MLP) / Feed-Forward Network:
        # Expands dimension 4x with non-linear activation (GELU), then projects back.
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        """Args:

            x: Input sequence of shape (batch, seq_len, embed_dim)
            attn_mask: Optional mask tensor (0 = keep, -inf = mask)
        Returns:
            Processed sequence of shape (batch, seq_len, embed_dim)
        """
        # Step 1: Pre-norm & Attention with residual connection
        h = self.norm1(x)

        if attn_mask is not None:
            attn_mask = attn_mask.to(dtype=h.dtype, device=h.device)

        attn_out, _ = self.attn(
            h, h, h,
            attn_mask=attn_mask,
        )
        x = x + attn_out  # Residual connection: keeps original information intact

        # Step 2: Pre-norm & MLP with residual connection
        h = self.norm2(x)
        x = x + self.mlp(h)  # Residual connection

        return x


def causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Constructs a causal upper-triangular mask of shape (seq_len, seq_len).

    Values:
        0.0  = Allowed to attend (diagonal and past tokens)
       -inf  = Forbidden from attending (future tokens)
    """
    future = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    return future.masked_fill(future == 1, float("-inf"))


def prefix_mask(n_prefix: int, n_text: int, device: torch.device) -> torch.Tensor:
    """Prefix Attention Mask for Vision-Language Models.

    Attention Rules:
      1. Prefix tokens (e.g. image patches) can all see each other bidirectionally (no mask).
      2. Prefix tokens CANNOT see any text tokens (text hasn't been generated yet).
      3. Text tokens can see ALL image prefix tokens and all earlier text tokens (causal).

    Matrix Layout (Allowed = 1, Masked = 0):
                 Prefix (Images)       Text (Words)
      Prefix:  [ 1  1  1  1  1 ]    [ 0  0  0  0 ]
      Text:    [ 1  1  1  1  1 ]    [ 1  0  0  0 ]
               [ 1  1  1  1  1 ]    [ 1  1  0  0 ]
               [ 1  1  1  1  1 ]    [ 1  1  1  0 ]
    """
    total = n_prefix + n_text
    allowed = torch.zeros(total, total, device=device)

    # Prefix tokens see each other
    allowed[:n_prefix, :n_prefix] = 1

    # Text tokens see prefix and past text
    if n_text:
        allowed[n_prefix:, :n_prefix] = 1
        allowed[n_prefix:, n_prefix:] = torch.tril(torch.ones(n_text, n_text, device=device))

    mask = torch.zeros(total, total, device=device)
    return mask.masked_fill(allowed == 0, float("-inf"))