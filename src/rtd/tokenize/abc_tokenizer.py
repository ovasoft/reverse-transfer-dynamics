"""
ABC-notation tokenizer -- see protocol document, Section 1 Step 3.

ABC notation is plain ASCII text, so this is a simple, from-scratch
whitespace/symbol-aware tokenizer with a vocab built from the training
corpus. It's intentionally minimal: good enough to get Phase 1 training
running end-to-end, not a claim that this is the optimal tokenization for
ABC. A BPE-based ABC tokenizer would likely do better and is a reasonable
early improvement for the team to make.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

_TOKEN_RE = re.compile(
    r"""
    \s+                 # whitespace runs
    | %.*                # comments
    | \[[A-Za-z]:[^\]]*\]  # inline fields e.g. [K:Cmaj]
    | \^\^?[A-Ga-g]      # double/single sharp
    | __?[A-Ga-g]        # double/single flat
    | =?[A-Ga-g][,']*    # note letter with octave marks
    | \d+                # numbers (durations)
    | /+                 # duration division
    | .                  # anything else, one char at a time
    """,
    re.VERBOSE,
)

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIAL_TOKENS = [PAD, BOS, EOS, UNK]


def split_abc(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text) if t.strip() != ""]


class ABCTokenizer:
    def __init__(self, vocab: dict[str, int] | None = None):
        self.token_to_id: dict[str, int] = vocab or {}
        self.id_to_token: dict[int, str] = {i: t for t, i in self.token_to_id.items()}

    @classmethod
    def build_from_corpus(cls, texts: list[str], *, min_freq: int = 1) -> "ABCTokenizer":
        freq: dict[str, int] = {}
        for text in texts:
            for tok in split_abc(text):
                freq[tok] = freq.get(tok, 0) + 1
        vocab = {t: i for i, t in enumerate(SPECIAL_TOKENS)}
        next_id = len(vocab)
        for tok, count in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0])):
            if count >= min_freq and tok not in vocab:
                vocab[tok] = next_id
                next_id += 1
        return cls(vocab)

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def encode(self, text: str, *, add_bos_eos: bool = True) -> list[int]:
        unk_id = self.token_to_id[UNK]
        ids = [self.token_to_id.get(t, unk_id) for t in split_abc(text)]
        if add_bos_eos:
            ids = [self.token_to_id[BOS]] + ids + [self.token_to_id[EOS]]
        return ids

    def decode(self, ids: list[int]) -> str:
        toks = [self.id_to_token.get(i, UNK) for i in ids]
        toks = [t for t in toks if t not in (PAD, BOS, EOS)]
        return "".join(toks)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.token_to_id, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "ABCTokenizer":
        vocab = json.loads(Path(path).read_text())
        return cls(vocab)
