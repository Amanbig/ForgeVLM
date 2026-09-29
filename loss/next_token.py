import torch.nn.functional as F


def next_token_loss(logits, token_ids):
    """Predict each text token from the position just before it.

    Image tokens, if present, occupy the front of `logits` and are left out of the loss.
    """

    n_text = token_ids.size(1)
    n_prefix = logits.size(1) - n_text

    if n_prefix == 0:
        predicting = logits[:, :-1, :]
        targets = token_ids[:, 1:]
    else:
        predicting = logits[:, n_prefix - 1:-1, :]
        targets = token_ids

    return F.cross_entropy(
        predicting.reshape(-1, predicting.size(-1)),
        targets.reshape(-1),
    )
