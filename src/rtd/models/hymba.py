"""
Hymba adapter -- STUB. See protocol document, Section 6, Step 1/3.

Reference implementation to build from: NVlabs/hymba on GitHub, which
includes a minimal nanoGPT-style reference implementation (confirmed
available as of this repo's creation -- see docs/DATA_SOURCES.md). The
released NVIDIA checkpoints (nvidia/Hymba-1.5B-Base on Hugging Face) are
1.5B scale; this project needs a from-scratch ~0.6B config trained on our
own ABC/MIDI/text data, not a fine-tune of the release.

Implementation notes:
  - Hymba's defining feature is parallel attention heads + SSM heads per
    layer, sharing input, plus learnable meta-tokens (see Section 6's
    architecture table). `get_attention_or_state()` should therefore
    return BOTH signals, not pick one: attention weights from the
    attention-head branch (same shape as TransformerAdapter's) AND the
    selective-scan parameters from the SSM-head branch (same shape as
    MambaAdapter's, once that's implemented) -- Section 6 Step 3 calls
    Hymba out specifically as "the only architecture where both an
    attention-based and a state-based read-out can be computed and
    compared directly", which is exactly why it's worth getting both
    signals out rather than simplifying to one.
  - Depends on MambaAdapter's selective-scan hook work being done first
    (or duplicated here) for the SSM-head branch.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from rtd.models.base import HiddenStateOutput, SequenceModelAdapter


class HymbaAdapter(SequenceModelAdapter):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "Hymba adapter not implemented in this starter repo -- see this "
            "file's module docstring. Build from github.com/NVlabs/hymba's "
            "reference implementation, configured down to ~0.6B params."
        )

    def forward(self, input_ids, **kwargs) -> HiddenStateOutput:
        raise NotImplementedError

    def get_attention_or_state(self, input_ids, **kwargs) -> Any:
        """Should return a dict: {'attention': ..., 'ssm_state': ...} --
        see module docstring. Don't collapse this to a single tensor."""
        raise NotImplementedError

    def num_params(self, *, non_embedding: bool = True) -> int:
        raise NotImplementedError

    def save_checkpoint(self, path: str | Path, *, step: int, extra: dict | None = None) -> None:
        raise NotImplementedError

    def load_checkpoint(self, path: str | Path) -> dict:
        raise NotImplementedError
