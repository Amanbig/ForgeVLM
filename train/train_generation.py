import torch

from data.tokens import ASSISTANT, BOI, BOS, EOI, USER, generation_example
from loss.next_token import next_token_loss
from model.generative_vlm import GenerativeVLM


def tiny_generative_vlm():
    return GenerativeVLM(
        text_vocab_size=64,
        num_visual_tokens=64,
        max_length=96,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
    )


def make_generation_batch(model: GenerativeVLM):
    # 4 text prompts asking for different images
    # e.g., Prompt 0: [10, 11] -> "red square", etc.
    prompts = [
        [10, 11],
        [12, 13],
        [14, 15],
        [16, 17],
    ]

    # Pre-generate discrete visual tokens using model's VQ-VAE on synthetic images
    images = torch.zeros(4, 3, 32, 32)
    images[0, 0, :16, :16] = 1.0  # red top-left
    images[1, 1, 16:, :16] = 1.0  # green bottom-left
    images[2, 2, :, 16:] = 1.0    # blue right-half
    images[3, 0, :, :] = 0.8      # yellow mix
    images[3, 1, :, :] = 0.8

    with torch.no_grad():
        visual_tokens = model.image_to_visual_tokens(images)  # (4, 64)

    # For training, use a subset of visual tokens (e.g. 16 tokens for speed)
    visual_seq = visual_tokens[:, :16].tolist()

    token_rows = []
    label_rows = []
    prompt_rows = []
    expected_rows = []

    for p, v in zip(prompts, visual_seq):
        token_ids, labels, prompt_len = generation_example(p, v)
        token_rows.append(token_ids)
        label_rows.append(labels)
        prompt_rows.append(token_ids[:prompt_len])
        expected_rows.append(v)

    return (
        images,
        torch.tensor(token_rows),
        torch.tensor(label_rows),
        torch.tensor(prompt_rows),
        torch.tensor(expected_rows),
    )


def main():
    torch.manual_seed(0)

    model = tiny_generative_vlm()
    images, token_ids, labels, prompt, expected = make_generation_batch(model)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"generative vlm parameters {n_params}")
    print(f"total vocab size (text + visual): {model.total_vocab}")
    print(f"sample token row: {token_ids[0].tolist()}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=0.0)

    first = None
    last = None

    for step in range(60):
        optimizer.zero_grad()
        logits = model(token_ids)
        loss = next_token_loss(logits, labels)
        loss.backward()
        optimizer.step()

        value = loss.item()
        if first is None:
            first = value
        last = value

        if step % 10 == 0 or step == 59:
            print(f"generative vlm step {step:03d}  loss {value:.4f}")

    print(f"generative vlm {first:.4f} -> {last:.4f}")

    if not last < first * 0.7:
        raise SystemExit(f"generative vlm loss did not fall: {first:.4f} -> {last:.4f}")

    # Generate image tokens autoregressively from the text prompt
    generated_tokens = model.generate_image_tokens(prompt, num_visual_tokens=expected.size(1))
    print(f"generated visual tokens: {generated_tokens[0].tolist()}")
    print(f"expected visual tokens:  {expected[0].tolist()}")

    if not torch.equal(generated_tokens, expected):
        raise SystemExit("autoregressive visual token generation did not match target")

    # Decode generated visual tokens into full RGB image
    # Note: we pad to 64 tokens if needed for the 8x8 grid
    full_tokens = torch.cat((generated_tokens, generated_tokens, generated_tokens, generated_tokens), dim=1)
    decoded_images = model.visual_tokens_to_image(full_tokens)
    assert decoded_images.shape == (4, 3, 32, 32)
    print("visual token generation and image decoding passed!")


if __name__ == "__main__":
    main()
