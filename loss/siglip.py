import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SigLIPLoss(nn.Module):
    """Sigmoid Loss for Language-Image Pre-training (SigLIP, Zhai et al., 2023).

    Why SigLIP instead of standard CLIP?
      Standard CLIP uses softmax across the whole batch row. This has two big drawbacks:
        1. It couples every pair in the batch, requiring expensive all-gather across GPUs.
        2. Softmax requires massive batch sizes (e.g. 32,768) to provide enough negatives.

      SigLIP replaces the softmax with independent binary sigmoid classifications for every
      single image-text pair in the batch:
        - Matching pair (image i, text i) is labeled +1 (YES, match!).
        - Mismatched pair (image i, text j) is labeled -1 (NO, not a match!).

    Loss Formula per pair:
      L(i, j) = -log sigmoid( y_ij * (exp(t) * (u_i @ v_j) + b) )
      where y_ij = +1 if i == j else -1.
    """

    def __init__(self):
        super().__init__()
        # Learnable temperature and bias parameters
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        # Negative bias starts unseen pairs on the "not a match" side of the sigmoid
        self.logit_bias = nn.Parameter(torch.tensor(-10.0))

    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """Args:

            image_features: L2-normalized image vectors of shape (batch, dim)
            text_features:  L2-normalized text vectors of shape (batch, dim)
        Returns:
            Scalar SigLIP loss
        """
        # Step 1: Pairwise cosine similarity matrix of shape (batch, batch)
        # Scaled by exp(logit_scale) and shifted by logit_bias
        logits = self.logit_scale.exp() * (image_features @ text_features.transpose(-2, -1))
        logits = logits + self.logit_bias

        # Step 2: Binary labels: +1 on diagonal (matches), -1 off diagonal (non-matches)
        # 2 * eye - 1 converts [[1, 0], [0, 1]] into [[+1, -1], [-1, +1]]
        n = logits.size(0)
        labels = 2 * torch.eye(n, device=logits.device) - 1

        # Step 3: Compute negative log-sigmoid: -log(1 / (1 + exp(-y * logits)))
        pair_loss = -F.logsigmoid(labels * logits)

        # Sum over texts for each image, then average over all images
        return pair_loss.sum(dim=1).mean()
