import torch

from data.tokens import ASSISTANT, EOS, IMAGE, chat_example
from loss.next_token import next_token_loss
from model.modern_vlm import ModernVLM


def tiny_modern_vlm():
    return ModernVLM(
        height=32,
        width=32,
        patch_size=8,
        in_channels=3,
        vision_width=64,
        vision_heads=4,
        vision_layers=2,
        vocab_size=64,
        max_length=64,
        lang_width=64,
        lang_heads=4,
        lang_layers=2,
        merge_size=2,
        image_token_id=IMAGE,
    )


def make_batch():
    question = [7, 8]
    answers = [
        [10, 11],
        [12, 13],
        [14, 15],
        [16, 17],
    ]

    token_rows = []
    label_rows = []
    prompt_rows = []
    expected_rows = []

    for answer in answers:
        token_ids, labels, prompt_len = chat_example(question, answer)
        token_rows.append(token_ids)
        label_rows.append(labels)
        prompt_rows.append(token_ids[:prompt_len])
        expected_rows.append(answer + [EOS])

    return (
        torch.tensor(token_rows),
        torch.tensor(label_rows),
        torch.tensor(prompt_rows),
        torch.tensor(expected_rows),
    )


def main():
    torch.manual_seed(0)

    model = tiny_modern_vlm()
    images = torch.randn(4, 3, 32, 32)
    token_ids, labels, prompt, expected = make_batch()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"modern vlm parameters: {n_params}")
    print(f"original patches: 16 -> merged visual tokens: {model.num_merged_tokens} (4x reduction)")

    logits = model(images, token_ids)
    targets = model.expand_labels(token_ids, labels)
    supervised = (targets != -100).sum(dim=1)

    print(f"logits shape: {tuple(logits.shape)} (only {logits.shape[1]} tokens per sequence!)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)

    first = None
    last = None

    for step in range(60):
        optimizer.zero_grad()
        loss = next_token_loss(model(images, token_ids), model.expand_labels(token_ids, labels))
        loss.backward()
        optimizer.step()

        value = loss.item()
        if first is None:
            first = value
        last = value

        if step % 10 == 0 or step == 59:
            print(f"modern vlm step {step:03d}  loss {value:.4f}")

    print(f"modern vlm {first:.4f} -> {last:.4f}")

    if not last < first * 0.7:
        raise SystemExit(f"modern vlm loss did not fall: {first:.4f} -> {last:.4f}")

    generated = model.generate(images, prompt, max_new_tokens=expected.size(1), eos_id=EOS)
    print(f"generated {generated.tolist()}")
    print(f"expected  {expected.tolist()}")

    if not torch.equal(generated, expected):
        raise SystemExit("modern vlm generation failed to reproduce answer")


if __name__ == "__main__":
    main()
