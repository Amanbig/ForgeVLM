import torch
import torch.nn as nn


class VectorQuantizer(nn.Module):
    """Vector Quantization layer (VQ-VAE / VQ-GAN).

    Maps continuous latent vectors into discrete codebook indices.
    Backpropagation is enabled via the Straight-Through Estimator (STE).
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z: torch.Tensor):
        """Args:

            z: continuous latents of shape (B, D, H, W) or (B, N, D)
        Returns:
            quantized: quantized tensor with same shape as z
            diff: VQ commitment and codebook loss
            indices: discrete codebook indices (B, H, W) or (B, N)
        """
        is_spatial = z.dim() == 4
        if is_spatial:
            # (B, D, H, W) -> (B, H, W, D) -> (B*H*W, D)
            b, d, h, w = z.shape
            z_flattened = z.permute(0, 2, 3, 1).contiguous().view(-1, d)
        else:
            b, n, d = z.shape
            z_flattened = z.contiguous().view(-1, d)

        # Compute squared Euclidean distances: ||z - e||^2 = ||z||^2 + ||e||^2 - 2*z*e^T
        dists = (
            torch.sum(z_flattened ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(z_flattened, self.embedding.weight.t())
        )

        # Find nearest codebook index
        encoding_indices = torch.argmin(dists, dim=1)
        quantized = self.embedding(encoding_indices)

        # VQ Losses: Codebook loss + Commitment loss
        loss_codebook = torch.mean((quantized - z_flattened.detach()) ** 2)
        loss_commitment = torch.mean((quantized.detach() - z_flattened) ** 2)
        vq_loss = loss_codebook + self.commitment_cost * loss_commitment

        # Straight-Through Estimator: copies gradients from quantized directly to z
        quantized = z_flattened + (quantized - z_flattened).detach()

        # Reshape back to input dimensions
        if is_spatial:
            quantized = quantized.view(b, h, w, d).permute(0, 3, 1, 2).contiguous()
            indices = encoding_indices.view(b, h, w)
        else:
            quantized = quantized.view(b, n, d)
            indices = encoding_indices.view(b, n)

        return quantized, vq_loss, indices

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Map discrete indices back to continuous embeddings."""
        return self.embedding(indices)
