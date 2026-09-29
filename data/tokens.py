# Special tokens reserved before any real word.
PAD = 0
IMAGE = 1
BOS = 2
USER = 3
ASSISTANT = 4
EOS = 5
VIDEO = 6
BOI = 7  # Beginning of Image / Visual sequence
EOI = 8  # End of Image / Visual sequence


def chat_example(question_ids, answer_ids, placeholder=IMAGE):
    """One turn: user text, one visual placeholder (IMAGE or VIDEO), then assistant answer.

    Labels on the prompt are -100, so the loss learns only the answer and the stop token.
    """
    prompt = [BOS, USER, placeholder, *question_ids, ASSISTANT]
    token_ids = prompt + list(answer_ids) + [EOS]
    labels = [-100] * len(prompt) + list(answer_ids) + [EOS]
    return token_ids, labels, len(prompt)


def generation_example(prompt_ids, visual_tokens):
    """Prompt asking model to generate an image:

    Prompt: [BOS, USER, *prompt_ids, ASSISTANT, BOI]
    Target: [*visual_tokens, EOI]
    """
    prompt = [BOS, USER, *prompt_ids, ASSISTANT, BOI]
    token_ids = prompt + list(visual_tokens) + [EOI]
    labels = [-100] * len(prompt) + list(visual_tokens) + [EOI]
    return token_ids, labels, len(prompt)
