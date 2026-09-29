import torch.nn.functional as F


def next_token_loss(logits, labels):
    """Autoregressive Next-Token Cross-Entropy Loss.

    How Next-Token Prediction Works:
      In causal language modeling, position t is trained to predict the token at position t + 1.

      Logits (input):   [ Token_0, Token_1, ..., Token_{T-2}, Token_{T-1} ]
      Labels (target):  [ Target_0, Target_1, Target_2, ..., Target_{T-1} ]

      Notice the 1-position shift:
        - The representation at Token_0 predicts Target_1.
        - The representation at Token_1 predicts Target_2.
        - ...
        - The representation at Token_{T-2} predicts Target_{T-1}.
        - The representation at Token_{T-1} has no target label, and Target_0 has no prior representation.

      Therefore:
        - `logits[:, :-1, :]`: All positions except the very last one.
        - `labels[:, 1:]`: All target tokens except the very first one.

      Ignore Index (-100):
        PyTorch's `F.cross_entropy` ignores any target equal to -100.
        Prompt tokens, image patch positions, and padding tokens are set to -100,
        so the loss ONLY trains on the assistant's answer!
    """
    return F.cross_entropy(
        logits[:, :-1, :].reshape(-1, logits.size(-1)),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )
