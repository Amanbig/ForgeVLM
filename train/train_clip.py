import torch

from loss.contrastive import contrastive_loss
from loss.siglip import SigLIPLoss
from model.model import CLIP


def tiny_clip():
    return CLIP(
        height=32,
        width=32,
        patch_size=8,
        in_channels=3,
        vision_width=64,
        vision_heads=4,
        vision_layers=2,
        vocab_size=64,
        context_length=16,
        text_width=64,
        text_heads=4,
        text_layers=2,
        projection_dim=32,
    )


def make_batch(batch_size, vocab_size, length):
    images = torch.randn(batch_size, 3, 32, 32)
    tokens = torch.randint(0, vocab_size - 1, (batch_size, length))
    tokens[:, -1] = vocab_size - 1
    return images, tokens


def run(name, steps, step_fn):
    first = None
    last = None

    for step in range(steps):
        loss = step_fn()

        if first is None:
            first = loss
        last = loss

        if step % 10 == 0 or step == steps - 1:
            print(f"{name} step {step:03d}  loss {loss:.4f}")

    print(f"{name} {first:.4f} -> {last:.4f}")

    if not last < first * 0.7:
        raise SystemExit(f"{name} loss did not fall: {first:.4f} -> {last:.4f}")


def main():
    torch.manual_seed(0)

    model = tiny_clip()
    images, tokens = make_batch(8, 64, 16)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"clip parameters {n_params}")

    with torch.no_grad():
        logits, _ = model(images, tokens)
    print(f"logits {tuple(logits.shape)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)

    def clip_step():
        optimizer.zero_grad()
        logits_per_image, logits_per_text = model(images, tokens)
        loss = contrastive_loss(logits_per_image, logits_per_text)
        loss.backward()
        optimizer.step()
        return loss.item()

    run("clip", 40, clip_step)

    torch.manual_seed(1)
    model = tiny_clip()
    images, tokens = make_batch(8, 64, 16)
    criterion = SigLIPLoss()
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=3e-3,
        weight_decay=0.0,
    )

    def siglip_step():
        optimizer.zero_grad()
        loss = criterion(model.encode_image(images), model.encode_text(tokens))
        loss.backward()
        optimizer.step()
        return loss.item()

    run("siglip", 80, siglip_step)


if __name__ == "__main__":
    main()
