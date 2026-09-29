# Ids reserved before any real word. The image id is a placeholder, not a patch.
IMAGE = 1
BOS = 2
USER = 3
ASSISTANT = 4
EOS = 5


def chat_example(question_ids, answer_ids):
    """One turn: user text, one image placeholder, then the assistant answer.

    Labels on the prompt are -100, so the loss learns the answer and the stop token.
    """

    prompt = [BOS, USER, IMAGE, *question_ids, ASSISTANT]
    token_ids = prompt + list(answer_ids) + [EOS]
    labels = [-100] * len(prompt) + list(answer_ids) + [EOS]
    return token_ids, labels, len(prompt)
