import torch
import torch.nn as nn

from embedding.vq import VectorQuantizer


class VQVAE(nn.Module):
    """Vector Quantized Variational AutoEncoder (VQ-VAE).

    Converts continuous RGB images into a discrete grid of integer codebook tokens,
    and decodes discrete codebook tokens back into reconstructed images.
    Used by modern multimodal generative models (e.g. Chameleon, Show-o, Parti).
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        num_embeddings: int = 128,
        commitment_cost: float = 0.25,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings

        # Encoder: downsamples image by 4x (e.g. 32x32 -> 8x8 grid of tokens)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, latent_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

        self.quantizer = VectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=latent_dim,
            commitment_cost=commitment_cost,
        )

        # Decoder: upsamples discrete latent grid 4x back to image pixels
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode_to_indices(self, x: torch.Tensor) -> torch.Tensor:
        """Args:

            x: (B, C, H, W) in [0, 1]
        Returns:
            indices: (B, grid_h, grid_w) integer tokens
        """
        z = self.encoder(x)
        _, _, indices = self.quantizer(z)
        return indices

    def decode_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Args:

            indices: (B, grid_h, grid_w) integer tokens
        Returns:
            reconstructed: (B, C, H, W)
        """
        # (B, H, W) -> (B, H, W, D) -> (B, D, H, W)
        z_q = self.quantizer.decode_indices(indices)
        if z_q.dim() == 4:
            z_q = z_q.permute(0, 3, 1, 2).contiguous()
        elif z_q.dim() == 3:
            # Flattened sequence (B, N, D)
            b, n, d = z_q.shape
            s = int(n ** 0.5)
            z_q = z_q.view(b, s, s, d).permute(0, 3, 1, 2).contiguous()
        return self.decoder(z_q)

    def forward(self, x: torch.Tensor):
        """Args:

            x: (B, C, H, W)
        Returns:
            reconstructed: (B, C, H, W)
            vq_loss: scalar commitment loss
            indices: (B, H_lat, W_lat)
        """
        z = self.encoder(x)
        quantized, vq_loss, indices = self.quantizer(z)
        reconstructed = self.decoder(quantized)
        return reconstructed, vq_loss, indices
