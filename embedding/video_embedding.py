import torch
import torch.nn as nn


class VideoPatchEmbedding(nn.Module):
    """3D Spatio-Temporal Tubelet Embedding for Video (ViViT / VideoMAE / TimeSformer).

    Turns a video of shape (batch, channels, time_frames, height, width) into a sequence
    of 3D tubelet tokens using a 3D convolution.
    """

    def __init__(
        self,
        num_frames: int,
        height: int,
        width: int,
        tubelet_time: int = 2,
        patch_size: int = 8,
        in_channels: int = 3,
        embed_dim: int = 64,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.tubelet_time = tubelet_time
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.grid_t = num_frames // tubelet_time
        self.grid_h = height // patch_size
        self.grid_w = width // patch_size
        self.num_tubelets = self.grid_t * self.grid_h * self.grid_w

        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=(tubelet_time, patch_size, patch_size),
            stride=(tubelet_time, patch_size, patch_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:

            x: (batch, in_channels, time, height, width)
        Returns:
            (batch, num_tubelets, embed_dim)
        """
        # x is (B, C, T, H, W)
        x = self.proj(x)  # (B, embed_dim, grid_t, grid_h, grid_w)
        x = x.flatten(2)  # (B, embed_dim, num_tubelets)
        x = x.transpose(1, 2)  # (B, num_tubelets, embed_dim)
        return x


class VideoCLIPEncoder(nn.Module):
    """Video Encoder using 3D tubelets and spatio-temporal position embeddings."""

    def __init__(
        self,
        num_frames: int,
        height: int,
        width: int,
        tubelet_time: int = 2,
        patch_size: int = 8,
        in_channels: int = 3,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()
        from transformer.transformer import TransformerBlock

        self.patch_embedding = VideoPatchEmbedding(
            num_frames=num_frames,
            height=height,
            width=width,
            tubelet_time=tubelet_time,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )

        num_tokens = self.patch_embedding.num_tubelets + 1
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))

        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embedding, std=0.02)

        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        tokens = self.patch_embedding(x)
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls, tokens), dim=1)
        x = x + self.pos_embedding

        for layer in self.layers:
            x = layer(x)

        return self.norm(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_tokens(x)[:, 0]
