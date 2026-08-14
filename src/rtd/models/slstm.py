"""
sLSTM (xLSTM) adapter -- STUB. See protocol document, Section 6, Step 1/3.

Install: pip install 'rtd[arch]'  (see pyproject.toml -- xlstm)

Implementation notes:
  - Use the official `xlstm` package's `xLSTMBlockStack` / `sLSTMBlock`
    (NX-AI/xlstm on GitHub) configured to use sLSTM blocks throughout
    (xLSTM also ships mLSTM blocks -- the protocol document specifically
    calls for sLSTM, with its scalar memory + exponential gating, so don't
    substitute mLSTM without updating the document too).
  - `get_attention_or_state()`: no attention weights exist. Return gate
    activations over time (forget-gate and input-gate values per step),
    per Section 6 Step 3's substitution table -- these are the closest
    recurrent analogue to an attention map, and Experiment 2.1's discovery
    pipeline (Step 3's "unsupervised functional signature per head",
    generalized to per-cell here) needs them.
  - sLSTM blocks are inherently sequential (no parallel scan the way Mamba
    has), so Phase 1/2 training will be slower per token than the other
    five architectures at the same token budget -- this is called out in
    Section 6 Step 7's compute-management notes; don't be surprised by it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from rtd.models.base import HiddenStateOutput, SequenceModelAdapter


class SLSTMAdapter(SequenceModelAdapter):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "sLSTM adapter not implemented in this starter repo -- see this "
            "file's module docstring. Requires the xlstm package "
            "(pip install 'rtd[arch]')."
        )

    def forward(self, input_ids, **kwargs) -> HiddenStateOutput:
        raise NotImplementedError

    def get_attention_or_state(self, input_ids, **kwargs) -> Any:
        raise NotImplementedError

    def num_params(self, *, non_embedding: bool = True) -> int:
        raise NotImplementedError

    def save_checkpoint(self, path: str | Path, *, step: int, extra: dict | None = None) -> None:
        raise NotImplementedError

    def load_checkpoint(self, path: str | Path) -> dict:
        raise NotImplementedError
