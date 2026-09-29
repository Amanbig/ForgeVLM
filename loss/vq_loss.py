import torch.nn.functional as F


def vqvae_loss(reconstructed, targets, vq_loss):
    """Reconstruction loss (MSE) combined with vector quantizer commitment loss."""
    recon_loss = F.mse_loss(reconstructed, targets)
    total_loss = recon_loss + vq_loss
    return total_loss, recon_loss
