"""
Mamba adapter -- STUB. See protocol document, Section 6, Step 1/3.

Not implemented in this starter repo: `mamba-ssm`'s selective-scan kernels
need a CUDA build, which this repo was scaffolded without GPU access to
verify. What's here is the interface shape and the specific implementation
notes needed to fill it in.

Install: pip install 'rtd[arch]'  (see pyproject.toml -- mamba-ssm)

Implementation notes:
  - Use `mamba_ssm.models.mixer_seq_simple.MambaLMHeadModel` (or build a
    stack of `mamba_ssm.modules.mamba_simple.Mamba` blocks directly if you
    want more control over hidden size / depth to hit the 0.6B param
    target from Section 1's compute-budget table).
  - `get_attention_or_state()` has NO attention weights to return (see
    base.SequenceModelAdapter's docstring) -- instead return the
    per-layer selective-scan parameters (the discretized Delta, and the
    B/C projection outputs). These aren't exposed by default; you'll need
    to register forward hooks on the Mamba mixer's internal `x_proj` /
    `dt_proj` submodules, or fork the block to return them. This is
    exactly the "analysis of the selective-scan Delta, B, C parameters"
    called out in Section 6 Step 3's substitution table -- get it right
    here and Experiment 2.5's Mamba row and Experiment 2.6's Mamba-branch
    ablation both depend on it.
  - `num_params(non_embedding=True)` should subtract the embedding AND the
    (usually tied) output projection, same logic as TransformerAdapter.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from rtd.models.base import HiddenStateOutput, SequenceModelAdapter


class MambaAdapter(SequenceModelAdapter):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "Mamba adapter not implemented in this starter repo -- see this "
            "file's module docstring for what's needed and why. Requires "
            "mamba-ssm (pip install 'rtd[arch]') and a CUDA-capable GPU to "
            "build the selective-scan kernels."
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
