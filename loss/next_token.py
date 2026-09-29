import torch.nn.functional as F


def next_token_loss(logits, labels):
    """Predict the next id. Positions marked -100 are left out of the loss.

    `labels` lines up with `logits`. The shift means position t is scored against labels[t + 1].
    """

    return F.cross_entropy(
        logits[:, :-1, :].reshape(-1, logits.size(-1)),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )
