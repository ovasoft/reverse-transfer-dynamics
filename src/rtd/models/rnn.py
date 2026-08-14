"""
Vanilla RNN adapter -- WEEK 1 EXERCISE. See WEEK1_EXERCISE.md and protocol
document, Section 6: "Plain tanh recurrence, no gating -- lower-bound
control" for H6 (capacity floor hypothesis).

This is deliberately the simplest architecture in the whole spectrum (no
gating, no attention, no selective state) -- implementing it is meant to
force real engagement with the SequenceModelAdapter interface (base.py)
without any architecture-specific complexity getting in the way. Every
other adapter in this package (transformer.py, and eventually mamba.py /
slstm.py / hymba.py) implements the exact same interface; if you can get
this one right, you understand what those need to do too.

Your task: implement every method below. `tests/test_rnn_exercise.py` has
the tests this needs to pass -- run it as you go, it's written to fail
loudly and specifically rather than silently.

Hints:
  - A single nn.RNN (or a small stack of them) plus an nn.Embedding and an
    nn.Linear output head is enough. Don't reach for LSTM/GRU -- the whole
    point of this architecture in the study is that it has NO gating.
  - `get_attention_or_state()` for an RNN has no attention weights and no
    gates -- per base.py's docstring, return the hidden-state trajectory
    itself (shape (batch, seq, hidden_dim) per layer). There is nothing
    richer to return, and that's the actual point: it's why RNN is the
    lower-bound control in Section 6's spectrum.
  - `output_hidden_states` isn't a built-in flag on nn.RNN the way it is on
    a HuggingFace model -- you'll need to either loop over timesteps
    yourself and collect states, or call the RNN multiple times, once per
    layer, forwarding each layer's full output sequence into the next.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from rtd.models.base import HiddenStateOutput, SequenceModelAdapter

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


class RNNAdapter(SequenceModelAdapter):
    def __init__(self, vocab_size: int, *, hidden_dim: int = 32, n_layers: int = 2, device: str = "cpu"):
        if not _TORCH_AVAILABLE:
            raise ImportError("torch not installed. Install with: pip install 'rtd[train]'")
        # TODO(Week 1 exercise): build nn.Embedding, nn.RNN (or a stack),
        # and an output nn.Linear(hidden_dim, vocab_size). Store what you
        # need on self so forward() below can use it.
        raise NotImplementedError("TODO(Week 1 exercise): implement __init__")

    def forward(self, input_ids, **kwargs) -> HiddenStateOutput:
        # TODO(Week 1 exercise): run the embedding + RNN stack, collect
        # per-layer hidden states into a tuple/list (see HiddenStateOutput's
        # docstring for the expected shape), and produce logits via the
        # output head. Return a HiddenStateOutput with attention_or_state
        # set to the hidden-state trajectory (see this module's docstring).
        raise NotImplementedError("TODO(Week 1 exercise): implement forward")

    def get_attention_or_state(self, input_ids, **kwargs) -> Any:
        # TODO(Week 1 exercise): this can likely just call forward() and
        # return its attention_or_state field -- see transformer.py's
        # get_attention_or_state for the pattern to follow.
        raise NotImplementedError("TODO(Week 1 exercise): implement get_attention_or_state")

    def num_params(self, *, non_embedding: bool = True) -> int:
        # TODO(Week 1 exercise): sum parameter counts; if non_embedding is
        # True, subtract the embedding table's parameters. See
        # transformer.py's num_params for the pattern.
        raise NotImplementedError("TODO(Week 1 exercise): implement num_params")

    def save_checkpoint(self, path: str | Path, *, step: int, extra: dict | None = None) -> None:
        raise NotImplementedError("TODO(Week 1 exercise): implement save_checkpoint")

    def load_checkpoint(self, path: str | Path) -> dict:
        raise NotImplementedError("TODO(Week 1 exercise): implement load_checkpoint")
