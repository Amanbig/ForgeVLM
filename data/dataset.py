import torch
from torch.utils.data import Dataset
from typing import List, Tuple

from data.tokenizer import CharTokenizer


def draw_shape(shape: str, color: Tuple[float, float, float], size: int = 32) -> torch.Tensor:
    """Generate a clean synthetic 32x32 RGB image tensor with a geometric shape."""
    img = torch.zeros(3, size, size)
    r, g, b = color

    cx, cy = size // 2, size // 2
    radius = size // 4

    y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")

    if shape == "circle":
        mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
    elif shape == "square":
        mask = (x >= cx - radius) & (x <= cx + radius) & (y >= cy - radius) & (y <= cy + radius)
    elif shape == "triangle":
        # simple upright triangle
        mask = (y >= cy - radius) & (y <= cy + radius) & (torch.abs(x - cx) <= (y - (cy - radius)) / 2)
    elif shape == "cross":
        w = radius // 2
        mask = ((torch.abs(x - cx) <= w) & (torch.abs(y - cy) <= radius)) | (
            (torch.abs(y - cy) <= w) & (torch.abs(x - cx) <= radius)
        )
    else:
        mask = torch.ones(size, size, dtype=torch.bool)

    img[0][mask] = r
    img[1][mask] = g
    img[2][mask] = b

    return img


class SyntheticMultimodalDataset(Dataset):
    """Dataset providing paired (image, token_ids, padding_mask) for CLIP/VLM training."""

    def __init__(self, tokenizer: CharTokenizer, max_length: int = 24, num_samples: int = 32):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        shapes = ["circle", "square", "triangle", "cross"]
        colors = [
            ("red", (1.0, 0.0, 0.0)),
            ("green", (0.0, 1.0, 0.0)),
            ("blue", (0.0, 0.0, 1.0)),
            ("yellow", (1.0, 1.0, 0.0)),
        ]

        for i in range(num_samples):
            color_name, rgb = colors[i % len(colors)]
            shape_name = shapes[(i // len(colors)) % len(shapes)]
            caption = f"a {color_name} {shape_name}"
            image = draw_shape(shape_name, rgb, size=32)
            self.samples.append((image, caption))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        image, caption = self.samples[idx]
        tokens = self.tokenizer.encode(caption, add_special_tokens=True)

        # Pad or truncate to max_length
        if len(tokens) < self.max_length:
            padding_mask = [1] * len(tokens) + [0] * (self.max_length - len(tokens))
            tokens = tokens + [self.tokenizer.pad_id] * (self.max_length - len(tokens))
        else:
            tokens = tokens[: self.max_length]
            tokens[-1] = self.tokenizer.eos_id
            padding_mask = [1] * self.max_length

        return (
            image,
            torch.tensor(tokens, dtype=torch.long),
            torch.tensor(padding_mask, dtype=torch.bool),
        )


class SyntheticVideoDataset(Dataset):
    """Dataset providing paired (video, token_ids) for Video VLM training."""

    def __init__(
        self,
        tokenizer: CharTokenizer,
        num_frames: int = 4,
        size: int = 32,
        max_length: int = 24,
        num_samples: int = 16,
    ):
        self.tokenizer = tokenizer
        self.num_frames = num_frames
        self.size = size
        self.max_length = max_length
        self.samples = []

        actions = [
            ("moving right", 1, 0),
            ("moving left", -1, 0),
            ("moving down", 0, 1),
            ("moving up", 0, -1),
        ]

        for i in range(num_samples):
            action_name, dx, dy = actions[i % len(actions)]
            caption = f"object {action_name}"

            # Create video frames (channels, time, height, width)
            video = torch.zeros(3, num_frames, size, size)
            for t in range(num_frames):
                cx = (size // 2) + dx * (t * 3)
                cy = (size // 2) + dy * (t * 3)
                y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
                mask = (x - cx) ** 2 + (y - cy) ** 2 <= (size // 6) ** 2
                video[0, t][mask] = 1.0
                video[1, t][mask] = 0.5

            self.samples.append((video, caption))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        video, caption = self.samples[idx]
        tokens = self.tokenizer.encode(caption, add_special_tokens=True)
        if len(tokens) < self.max_length:
            tokens = tokens + [self.tokenizer.pad_id] * (self.max_length - len(tokens))
        else:
            tokens = tokens[: self.max_length]
            tokens[-1] = self.tokenizer.eos_id

        return video, torch.tensor(tokens, dtype=torch.long)
