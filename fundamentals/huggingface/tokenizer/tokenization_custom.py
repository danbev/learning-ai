from transformers import PreTrainedTokenizer
from typing import List, Optional

class CustomTokenizer(PreTrainedTokenizer):

    def __init__(self, vocab_size=100, auto_map=None, **kwargs):
        print(f"Initializing CustomTokenizer with vocab_size={vocab_size} and kwargs={kwargs}")
        # Defaults if not provided via kwargs (e.g., when loading from config)
        default_pad = "[PAD]"
        default_unk = "[UNK]"
        default_bos = "[BOS]"
        default_eos = "[EOS]"

        # Pull possibly-present values from kwargs FIRST to avoid duplicates
        pad_token = kwargs.pop("pad_token", default_pad)
        unk_token = kwargs.pop("unk_token", default_unk)
        bos_token = kwargs.pop("bos_token", default_bos)
        eos_token = kwargs.pop("eos_token", default_eos)

        self._vocab_size_value = vocab_size
        self._pad_token = pad_token
        self._unk_token = unk_token
        self._bos_token = bos_token
        self._eos_token = eos_token

        # Build vocab: special tokens + printable ASCII chars
        special_tokens = [self._pad_token, self._unk_token, self._bos_token, self._eos_token]
        print(f"{special_tokens=}")
        chars = [chr(i) for i in range(32, 127)]
        print(chars)
        vocab_list = special_tokens + chars[: max(0, vocab_size - len(special_tokens))]
        print(f"{vocab_list=}")

        self._vocab = {tok: idx for idx, tok in enumerate(vocab_list)}
        self._reverse_vocab = {idx: tok for tok, idx in self._vocab.items()}

        # Optional: advertise class name (helps some HF versions)
        self.tokenizer_class = self.__class__.__name__

        super().__init__(
            pad_token=self._pad_token,
            unk_token=self._unk_token,
            bos_token=self._bos_token,
            eos_token=self._eos_token,
            auto_map=auto_map,
            **kwargs,  # now safe — no duplicate special-token keys left
        )

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)  # Return actual vocab length

    def get_vocab(self):
        return self._vocab.copy()

    # This is what decides how to split text into tokens.
    def _tokenize(self, text: str) -> List[str]:
        return list(text)

    # This function will get called when tokenizing a string. It is what allows
    # the tokenizer to convert from token to ID.
    def _convert_token_to_id(self, token: str) -> int:
        # _vocab is a dict mapping token to ID, and we are specifying a default
        # value if the token is not found which is the ID for the unk_token.
        return self._vocab.get(token, self._vocab[self._unk_token])

    def _convert_id_to_token(self, index: int) -> str:
        return self._reverse_vocab.get(index, self._unk_token)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return "".join(tokens)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> tuple:
        import os
        import json

        if filename_prefix is None:
            filename_prefix = ""

        vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + "vocab.json"
        )

        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self._vocab, f, ensure_ascii=False, indent=2)

        return (vocab_file,)
