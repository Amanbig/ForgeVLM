import torch

from loss.vq_loss import vqvae_loss
from model.vqvae import VQVAE


def tiny_vqvae():
    return VQVAE(
        in_channels=3,
        hidden_dim=32,
        latent_dim=16,
        num_embeddings=64,
        commitment_cost=0.25,
    )


def make_image_batch(batch_size=4):
    images = torch.zeros(batch_size, 3, 32, 32)
    # Simple synthetic colored patterns with distinct blocks
    for i in range(batch_size):
        images[i, 0, :16, :16] = (i + 1) / (batch_size + 1)
        images[i, 1, 16:, :16] = 1.0 - (i / batch_size)
        images[i, 2, :, 16:] = 0.5
    return images


def main():
    torch.manual_seed(0)

    model = tiny_vqvae()
    images = make_image_batch(4)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"vqvae parameters {n_params}")

    indices = model.encode_to_indices(images)
    print(f"encoded discrete visual tokens shape: {tuple(indices.shape)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)

    first = None
    last = None

    for step in range(50):
        optimizer.zero_grad()
        recon, vq_l, _ = model(images)
        loss, recon_l = vqvae_loss(recon, images, vq_l)
        loss.backward()
        optimizer.step()

        value = loss.item()
        if first is None:
            first = value
        last = value

        if step % 10 == 0 or step == 49:
            print(f"vqvae step {step:03d}  total loss {value:.4f}  recon mse {recon_l.item():.4f}")

    print(f"vqvae {first:.4f} -> {last:.4f}")

    if not last < first * 0.7:
        raise SystemExit(f"vqvae loss did not fall: {first:.4f} -> {last:.4f}")

    # Test reconstruction directly from discrete codebook indices
    reconstructed = model.decode_from_indices(indices)
    assert reconstructed.shape == images.shape, "reconstructed shape mismatch"
    print("reconstruction from discrete indices passed")


if __name__ == "__main__":
    main()
