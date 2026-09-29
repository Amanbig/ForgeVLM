import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SigLIPLoss(nn.Module):
    """Pairwise sigmoid loss. Every image-text pair is its own yes/no decision."""

    def __init__(self):
        super().__init__()

        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.logit_bias = nn.Parameter(torch.tensor(-10.0))

    def forward(self, image_features, text_features):
        logits = self.logit_scale.exp() * (image_features @ text_features.transpose(-2, -1))
        logits = logits + self.logit_bias

        n = logits.size(0)
        labels = 2 * torch.eye(n, device=logits.device) - 1

        pair_loss = -F.logsigmoid(labels * logits)
        return pair_loss.sum(dim=1).mean()
