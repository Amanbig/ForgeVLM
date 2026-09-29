import torch

from data.tokens import ASSISTANT, EOS, VIDEO, chat_example
from loss.next_token import next_token_loss
from model.video_vlm import VideoVLM


def tiny_video_vlm():
    return VideoVLM(
        num_frames=4,
        height=32,
        width=32,
        tubelet_time=2,
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
        video_token_id=VIDEO,
    )


def make_video_batch():
    # 4 videos, each with a different temporal pattern (e.g. motion or color evolution)
    # Question asks: "What action happens in the video?" (tokens [7, 8])
    question = [7, 8]
    answers = [
        [10, 11],  # "moving right"
        [12, 13],  # "moving left"
        [14, 15],  # "jumping up"
        [16, 17],  # "falling down"
    ]

    token_rows = []
    label_rows = []
    prompt_rows = []
    expected_rows = []

    for answer in answers:
        token_ids, labels, prompt_len = chat_example(question, answer, placeholder=VIDEO)
        token_rows.append(token_ids)
        label_rows.append(labels)
        prompt_rows.append(token_ids[:prompt_len])
        expected_rows.append(answer + [EOS])

    # Generate synthetic video tensors: (batch=4, channels=3, time=4, height=32, width=32)
    videos = torch.zeros(4, 3, 4, 32, 32)
    for b in range(4):
        for t in range(4):
            # Create distinguishable temporal dynamics per video
            videos[b, :, t, (b * 8 + t * 2) % 32, :] = 1.0

    return (
        videos,
        torch.tensor(token_rows),
        torch.tensor(label_rows),
        torch.tensor(prompt_rows),
        torch.tensor(expected_rows),
    )


def main():
    torch.manual_seed(0)

    model = tiny_video_vlm()
    videos, token_ids, labels, prompt, expected = make_video_batch()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"video vlm parameters {n_params}")
    print(f"video batch shape {tuple(videos.shape)}")
    print(f"chat row {token_ids[0].tolist()}")

    logits = model(videos, token_ids)
    targets = model.expand_labels(token_ids, labels)
    supervised = (targets != -100).sum(dim=1)

    print(f"logits {tuple(logits.shape)}  supervised {supervised.tolist()}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)

    first = None
    last = None

    for step in range(60):
        optimizer.zero_grad()
        loss = next_token_loss(model(videos, token_ids), model.expand_labels(token_ids, labels))
        loss.backward()
        optimizer.step()

        value = loss.item()
        if first is None:
            first = value
        last = value

        if step % 10 == 0 or step == 59:
            print(f"video vlm step {step:03d}  loss {value:.4f}")

    print(f"video vlm {first:.4f} -> {last:.4f}")

    if not last < first * 0.7:
        raise SystemExit(f"video vlm loss did not fall: {first:.4f} -> {last:.4f}")

    generated = model.generate(videos, prompt, max_new_tokens=expected.size(1), eos_id=EOS)
    print(f"generated {generated.tolist()}")
    print(f"expected  {expected.tolist()}")

    if not torch.equal(generated, expected):
        raise SystemExit("video generation did not reproduce the answer for each video")


if __name__ == "__main__":
    main()
