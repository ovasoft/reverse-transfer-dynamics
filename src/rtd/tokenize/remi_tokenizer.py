"""
MIDI -> REMI tokenization wrapper -- see protocol document, Section 1
Step 1 ("parse with MidiTok using a REMI tokenizer config").

Thin wrapper around the `miditok` package so the rest of the codebase
depends on this module's interface rather than on miditok's API directly
(miditok's API has changed across major versions before; isolating it here
means an upgrade only touches one file).
"""
from __future__ import annotations

from pathlib import Path

try:
    from miditok import REMI, TokenizerConfig
    from symusic import Score
    _MIDITOK_AVAILABLE = True
except ImportError:
    _MIDITOK_AVAILABLE = False


DEFAULT_CONFIG = dict(
    pitch_range=(21, 109),
    beat_res={(0, 4): 8, (4, 12): 4},
    num_velocities=32,
    use_chords=False,
    use_rests=True,
    use_tempos=True,
    use_time_signatures=True,
    use_programs=False,
)


class RemiTokenizerWrapper:
    def __init__(self, config_overrides: dict | None = None):
        if not _MIDITOK_AVAILABLE:
            raise ImportError(
                "miditok/symusic not installed. Install with: pip install miditok"
            )
        cfg = {**DEFAULT_CONFIG, **(config_overrides or {})}
        self.tokenizer = REMI(TokenizerConfig(**cfg))

    def encode_file(self, midi_path: str | Path) -> list[int]:
        score = Score(str(midi_path))
        tok_seq = self.tokenizer(score)
        # miditok returns one TokSequence per track for multi-track REMI;
        # flatten and concatenate -- fine for Phase 1 pretraining where we
        # just need a token stream, revisit if per-track structure matters
        # later (e.g. for M3 dependency-distance work across instruments).
        if isinstance(tok_seq, list):
            ids: list[int] = []
            for seq in tok_seq:
                ids.extend(seq.ids)
            return ids
        return tok_seq.ids

    def decode(self, ids: list[int]):
        return self.tokenizer(ids)

    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer.vocab)

    def save_params(self, path: str | Path) -> None:
        self.tokenizer.save(str(path))

    @classmethod
    def from_saved(cls, path: str | Path) -> "RemiTokenizerWrapper":
        if not _MIDITOK_AVAILABLE:
            raise ImportError("miditok/symusic not installed.")
        obj = cls.__new__(cls)
        obj.tokenizer = REMI(params=str(path))
        return obj
