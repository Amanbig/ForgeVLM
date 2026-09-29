import torch
import torch.nn.functional as F


def contrastive_loss(logits_per_image: torch.Tensor, logits_per_text: torch.Tensor) -> torch.Tensor:
    """Symmetric InfoNCE Contrastive Loss (CLIP).

    How it works:
      Given a batch of N (image, text) pairs:
      The diagonal entries (0,0), (1,1), ..., (N-1, N-1) are positive pairs.
      All off-diagonal entries (i, j) where i != j are negative pairs.

      1. Image-to-Text Loss:
         For each image row i, treat the N texts as classes and predict class i
         using CrossEntropyLoss with target = [0, 1, 2, ..., N-1].
      2. Text-to-Image Loss:
         For each text row j, treat the N images as classes and predict class j
         using CrossEntropyLoss with target = [0, 1, 2, ..., N-1].
      3. Total loss is the average of both directions.
    """
    # targets: [0, 1, 2, ..., batch_size - 1]
    targets = torch.arange(logits_per_image.size(0), device=logits_per_image.device)

    loss_image = F.cross_entropy(logits_per_image, targets)
    loss_text = F.cross_entropy(logits_per_text, targets)

    return (loss_image + loss_text) / 2
