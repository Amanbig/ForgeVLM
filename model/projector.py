import math
import torch
import torch.nn as nn


class Projector(nn.Module):
    """Linear / MLP projector from vision width to language width (LLaVA-1.0 style)."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SpatialMergeProjector(nn.Module):
    """2x2 Spatial Merging / Pixel Unshuffle Projector (LLaVA-NeXT, Qwen2-VL, InternVL style).

    Takes a 2D spatial grid of patch tokens (B, H, W, D), packs each 2x2 neighborhood
    into a single token of width 4*D, and projects to language width.
    Reduces visual token count by 4x without discarding visual information.
    """

    def __init__(self, in_dim: int, out_dim: int, merge_size: int = 2):
        super().__init__()
        self.merge_size = merge_size
        self.in_dim = in_dim
        self.out_dim = out_dim

        merged_dim = in_dim * (merge_size * merge_size)
        self.mlp = nn.Sequential(
            nn.LayerNorm(merged_dim),
            nn.Linear(merged_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor, grid_h: int = None, grid_w: int = None) -> torch.Tensor:
        """Args:

        x: (batch, num_patches, in_dim) or (batch, H, W, in_dim)
        grid_h, grid_w: spatial dimensions of the patch grid
        """
        if x.dim() == 3:
            b, n, d = x.shape
            if grid_h is None or grid_w is None:
                # Assume square grid if not specified
                s = int(math.isqrt(n))
                if s * s != n:
                    raise ValueError(f"num_patches {n} is not a square grid, provide grid_h and grid_w")
                grid_h = grid_w = s
            x = x.view(b, grid_h, grid_w, d)
        else:
            b, grid_h, grid_w, d = x.shape

        m = self.merge_size
        if grid_h % m != 0 or grid_w % m != 0:
            raise ValueError(f"Grid ({grid_h}, {grid_w}) must be divisible by merge_size {m}")

        # Reshape to group 2x2 patches: (B, H//2, 2, W//2, 2, D) -> (B, H//2, W//2, 4*D)
        new_h = grid_h // m
        new_w = grid_w // m
        x = x.view(b, new_h, m, new_w, m, d)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(b, new_h * new_w, m * m * d)

        return self.mlp(x)


class PerceiverResampler(nn.Module):
    """Perceiver Resampler / Q-Former (Flamingo, BLIP-2 style).

    Compresses an arbitrary sequence of visual tokens (from high-res crops or video frames)
    into a fixed number of latent visual tokens using cross-attention.
    """

    def __init__(
        self,
        num_queries: int,
        in_dim: int,
        out_dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.queries = nn.Parameter(torch.randn(num_queries, out_dim) * 0.02)

        self.in_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "cross_attn": nn.MultiheadAttention(out_dim, num_heads, batch_first=True),
                "cross_norm": nn.LayerNorm(out_dim),
                "self_attn": nn.MultiheadAttention(out_dim, num_heads, batch_first=True),
                "self_norm": nn.LayerNorm(out_dim),
                "mlp": nn.Sequential(
                    nn.Linear(out_dim, 4 * out_dim),
                    nn.GELU(),
                    nn.Linear(4 * out_dim, out_dim),
                ),
                "mlp_norm": nn.LayerNorm(out_dim),
            })
            for _ in range(num_layers)
        ])

    def forward(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        """Args:

        visual_tokens: (batch, num_visual_tokens, in_dim)
        Returns:
            (batch, num_queries, out_dim)
        """
        b = visual_tokens.size(0)
        kv = self.in_proj(visual_tokens)
        q = self.queries.unsqueeze(0).expand(b, -1, -1)

        for layer in self.layers:
            # Cross-attention: queries attend to visual tokens
            h_q = layer["cross_norm"](q)
            cross_out, _ = layer["cross_attn"](query=h_q, key=kv, value=kv)
            q = q + cross_out

            # Self-attention: queries interact with each other
            h_q = layer["self_norm"](q)
            self_out, _ = layer["self_attn"](query=h_q, key=h_q, value=h_q)
            q = q + self_out

            # MLP
            h_q = layer["mlp_norm"](q)
            q = q + layer["mlp"](h_q)

        return q
