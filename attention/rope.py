import torch
import torch.nn as nn


def precompute_freqs_cis(dim: int, max_seq_len: int, theta: float = 10000.0):
    """Precompute complex frequencies for 1D RoPE."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    # Return (cos, sin) pairs
    return torch.cos(freqs), torch.sin(freqs)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions: [-x2, x1, -x4, x3, ...]."""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rope_1d(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply 1D Rotary Position Embedding to (batch, heads, seq_len, head_dim)."""
    # Repeat cos and sin across alternate elements to match head_dim
    cos = cos.repeat_interleave(2, dim=-1)[..., :x.shape[-1]]
    sin = sin.repeat_interleave(2, dim=-1)[..., :x.shape[-1]]
    return (x * cos) + (rotate_half(x) * sin)



class RotaryEmbedding(nn.Module):
    """1D Rotary Position Embedding (RoPE) used in modern LLMs (Llama, Mistral)."""

    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        cos, sin = precompute_freqs_cis(dim, max_seq_len, theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int = None) -> torch.Tensor:
        # x: (batch, heads, seq_len, head_dim)
        if seq_len is None:
            seq_len = x.shape[-2]
        cos = self.cos[:seq_len, :].to(dtype=x.dtype, device=x.device)
        sin = self.sin[:seq_len, :].to(dtype=x.dtype, device=x.device)
        return apply_rope_1d(x, cos, sin)


class MultimodalRoPE(nn.Module):
    """2D and 3D Rotary Position Embedding used in modern VLMs (e.g. Qwen2-VL).

    Splits head_dim into:
      - 2D: (height_dim, width_dim) for images
      - 3D: (time_dim, height_dim, width_dim) for videos
    """

    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward_2d(self, x: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
        """Apply 2D spatial RoPE to image patches: x shape (batch, heads, grid_h * grid_w, head_dim)."""
        dim_h = self.dim // 2
        dim_w = self.dim - dim_h

        # Precompute frequencies for height and width
        cos_h, sin_h = precompute_freqs_cis(dim_h, grid_h, self.theta)
        cos_w, sin_w = precompute_freqs_cis(dim_w, grid_w, self.theta)

        # Build 2D meshgrid of coordinates
        h_idx = torch.arange(grid_h, device=x.device)
        w_idx = torch.arange(grid_w, device=x.device)
        grid_y, grid_x = torch.meshgrid(h_idx, w_idx, indexing="ij")
        grid_y = grid_y.reshape(-1)
        grid_x = grid_x.reshape(-1)

        # Gather per-patch cos/sin
        cos_h = cos_h[grid_y].to(dtype=x.dtype, device=x.device)
        sin_h = sin_h[grid_y].to(dtype=x.dtype, device=x.device)
        cos_w = cos_w[grid_x].to(dtype=x.dtype, device=x.device)
        sin_w = sin_w[grid_x].to(dtype=x.dtype, device=x.device)

        # Split x into height and width components
        xh = x[..., :dim_h]
        xw = x[..., dim_h:]

        xh_rot = apply_rope_1d(xh, cos_h, sin_h)
        xw_rot = apply_rope_1d(xw, cos_w, sin_w)

        return torch.cat((xh_rot, xw_rot), dim=-1)

    def forward_3d(self, x: torch.Tensor, grid_t: int, grid_h: int, grid_w: int) -> torch.Tensor:
        """Apply 3D spatio-temporal RoPE to video tubelets: x shape (batch, heads, T * H * W, head_dim)."""
        dim_t = 2 * (self.dim // 6)
        dim_h = 2 * (self.dim // 6)
        dim_w = self.dim - dim_t - dim_h

        cos_t, sin_t = precompute_freqs_cis(dim_t, grid_t, self.theta)
        cos_h, sin_h = precompute_freqs_cis(dim_h, grid_h, self.theta)
        cos_w, sin_w = precompute_freqs_cis(dim_w, grid_w, self.theta)

        t_idx = torch.arange(grid_t, device=x.device)
        h_idx = torch.arange(grid_h, device=x.device)
        w_idx = torch.arange(grid_w, device=x.device)
        grid_time, grid_y, grid_x = torch.meshgrid(t_idx, h_idx, w_idx, indexing="ij")

        grid_time = grid_time.reshape(-1)
        grid_y = grid_y.reshape(-1)
        grid_x = grid_x.reshape(-1)

        cos_t = cos_t[grid_time].to(dtype=x.dtype, device=x.device)
        sin_t = sin_t[grid_time].to(dtype=x.dtype, device=x.device)
        cos_h = cos_h[grid_y].to(dtype=x.dtype, device=x.device)
        sin_h = sin_h[grid_y].to(dtype=x.dtype, device=x.device)
        cos_w = cos_w[grid_x].to(dtype=x.dtype, device=x.device)
        sin_w = sin_w[grid_x].to(dtype=x.dtype, device=x.device)

        xt = x[..., :dim_t]
        xh = x[..., dim_t:dim_t + dim_h]
        xw = x[..., dim_t + dim_h:]

        xt_rot = apply_rope_1d(xt, cos_t, sin_t)
        xh_rot = apply_rope_1d(xh, cos_h, sin_h)
        xw_rot = apply_rope_1d(xw, cos_w, sin_w)

        return torch.cat((xt_rot, xh_rot, xw_rot), dim=-1)
