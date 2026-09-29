import torch

from loss.next_token import next_token_loss
from model.vlm import VLM


def tiny_vlm():
    return VLM(
        height=32,
        width=32,
        patch_size=8,
        in_channels=3,
        vision_width=64,
        vision_heads=4,
        vision_layers=2,
        vocab_size=64,
        max_length=16,
        lang_width=64,
        lang_heads=4,
        lang_layers=2,
    )


def main():
    torch.manual_seed(0)

    model = tiny_vlm()
    images = torch.randn(4, 3, 32, 32)
    tokens = torch.randint(0, 64, (4, 12))

    n_params = sum(p.numel() for p in model.parameters())
    n_patches = model.vision.patch_embedding.num_patches
    print(f"vlm parameters {n_params}")

    with torch.no_grad():
        logits = model(images, tokens)
    print(f"logits {tuple(logits.shape)}  patches {n_patches}")

    if logits.shape[1] != n_patches + tokens.shape[1]:
        raise SystemExit(
            f"expected sequence {n_patches + tokens.shape[1]}, got {logits.shape[1]}"
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)

    first = None
    last = None

    for step in range(40):
        optimizer.zero_grad()
        loss = next_token_loss(model(images, tokens), tokens)
        loss.backward()
        optimizer.step()

        value = loss.item()
        if first is None:
            first = value
        last = value

        if step % 10 == 0 or step == 39:
            print(f"vlm step {step:03d}  loss {value:.4f}")

    print(f"vlm {first:.4f} -> {last:.4f}")

    if not last < first * 0.7:
        raise SystemExit(f"vlm loss did not fall: {first:.4f} -> {last:.4f}")


if __name__ == "__main__":
    main()
