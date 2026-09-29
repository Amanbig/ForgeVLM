import math
import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention (MHA).

    Why Multi-Head?
    A single attention head can only focus on one kind of relationship at a time
    (e.g., syntactic structure). Splitting the hidden dimension into `num_heads`
    allows different heads to learn different things in parallel:
      - Head 1 might focus on visual color matching.
      - Head 2 might focus on object boundaries.
      - Head 3 might focus on positional proximity.

    Supports:
      - Self-Attention: x attends to x (when context is None)
      - Cross-Attention: x queries attend to context keys/values (used in Resamplers/Decoders)
    """

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Linear projections for Query, Key, and Value
        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)

        # Output projection combining all heads back into embed_dim
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor = None, attn_mask: torch.Tensor = None) -> torch.Tensor:
        """Args:

            x: Query sequence of shape (batch, seq_len_q, embed_dim)
            context: Optional Key/Value source sequence (batch, seq_len_kv, embed_dim).
                     If None, performs self-attention where context = x.
            attn_mask: Optional mask added to attention scores before softmax.
        Returns:
            Output sequence of shape (batch, seq_len_q, embed_dim)
        """
        # Step 1: Select source for keys and values
        kv = x if context is None else context

        # Step 2: Linear projections
        # q: (batch, seq_len_q, embed_dim)
        # k, v: (batch, seq_len_kv, embed_dim)
        q = self.q(x)
        k = self.k(kv)
        v = self.v(kv)

        b, seq_q, _ = q.shape
        _, seq_k, _ = k.shape

        # Step 3: Split into multiple heads and transpose for matrix multiplication
        # (batch, seq, embed_dim) -> (batch, seq, num_heads, head_dim) -> (batch, num_heads, seq, head_dim)
        q = q.view(b, seq_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, seq_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, seq_k, self.num_heads, self.head_dim).transpose(1, 2)

        # Step 4: Scaled dot-product attention per head
        # (batch, num_heads, seq_q, head_dim) @ (batch, num_heads, head_dim, seq_k)
        # -> shape: (batch, num_heads, seq_q, seq_k)
        score = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Step 5: Add attention mask if provided
        if attn_mask is not None:
            score = score + attn_mask.to(dtype=score.dtype, device=score.device)

        # Step 6: Softmax over the last dimension (keys) to get attention probabilities
        attn = torch.softmax(score, dim=-1)

        # Step 7: Multiply attention probabilities by values
        # (batch, num_heads, seq_q, seq_k) @ (batch, num_heads, seq_k, head_dim)
        # -> shape: (batch, num_heads, seq_q, head_dim)
        out = attn @ v

        # Step 8: Concatenate heads back together
        # (batch, num_heads, seq_q, head_dim) -> (batch, seq_q, num_heads, head_dim) -> (batch, seq_q, embed_dim)
        out = out.transpose(1, 2).contiguous().view(b, seq_q, self.embed_dim)

        # Step 9: Final linear projection
        return self.out(out)