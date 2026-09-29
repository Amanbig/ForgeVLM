import torch
import torch.nn.functional as F


def contrastive_loss(logits_per_image, logits_per_text):
    """Symmetric cross-entropy. Row i of the image logits should pick text i."""

    targets = torch.arange(logits_per_image.size(0), device=logits_per_image.device)
    loss_image = F.cross_entropy(logits_per_image, targets)
    loss_text = F.cross_entropy(logits_per_text, targets)
    return (loss_image + loss_text) / 2
