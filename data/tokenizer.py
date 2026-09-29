from typing import List, Dict, Optional
from data.tokens import PAD, IMAGE, BOS, USER, ASSISTANT, EOS, VIDEO, BOI, EOI


class CharTokenizer:
    """A clean, from-scratch character-level tokenizer with multimodal special tokens.

    Maps text characters to token IDs and back.
    Special tokens are mapped to low IDs, and the vocabulary is self-contained.
    """

    def __init__(self, vocab_chars: Optional[str] = None):
        # Default vocabulary: ASCII printable characters
        if vocab_chars is None:
            vocab_chars = (
                " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?:;'-_()/"
            )

        self.special_tokens = {
            "<pad>": PAD,
            "<image>": IMAGE,
            "<bos>": BOS,
            "<user>": USER,
            "<assistant>": ASSISTANT,
            "<eos>": EOS,
            "<video>": VIDEO,
            "<boi>": BOI,
            "<eoi>": EOI,
        }

        self.id_to_token: Dict[int, str] = {v: k for k, v in self.special_tokens.items()}
        self.token_to_id: Dict[str, int] = dict(self.special_tokens)

        # Assign character IDs starting after the special tokens
        next_id = max(self.special_tokens.values()) + 1
        for ch in vocab_chars:
            if ch not in self.token_to_id:
                self.token_to_id[ch] = next_id
                self.id_to_token[next_id] = ch
                next_id += 1

        self.vocab_size = next_id
        self.pad_id = PAD
        self.eos_id = EOS
        self.image_id = IMAGE
        self.video_id = VIDEO

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Encode a string into token IDs."""
        ids = []
        if add_special_tokens:
            ids.append(BOS)

        for ch in text:
            if ch in self.token_to_id:
                ids.append(self.token_to_id[ch])
            else:
                # Fallback to space if unknown character
                ids.append(self.token_to_id.get(" ", self.pad_id))

        if add_special_tokens:
            ids.append(EOS)

        return ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back into a string."""
        chars = []
        for tid in token_ids:
            if tid in self.id_to_token:
                val = self.id_to_token[tid]
                if skip_special_tokens and val.startswith("<") and val.endswith(">"):
                    continue
                chars.append(val)
        return "".join(chars)
