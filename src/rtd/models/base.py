"""
Common architecture adapter interface -- see protocol document, Section 6
(Experiment 2.5), Step 1, and the note in the codebase-planning discussion:
every architecture (Transformer, GPT2, Hymba, Mamba, sLSTM, RNN) needs to
expose the SAME surface so Experiments 2.1, 2.2, 2.5, 2.6 can be written
once against this interface instead of once per architecture.

Requires torch (see pyproject.toml [train] extra). Import is deferred so
the rest of the `rtd` package works without torch installed.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class HiddenStateOutput:
    """Uniform container for whatever a forward pass produces, regardless
    of architecture. `attention_or_state` is architecture-dependent (see
    docstring on get_attention_or_state) and may be None."""

    hidden_states: Any  # tuple/list of tensors, one per layer, shape (batch, seq, dim)
    logits: Any
    attention_or_state: Any | None = None


class SequenceModelAdapter(ABC):
    """Subclass this once per architecture. See src/rtd/models/transformer.py
    for a complete reference implementation, and src/rtd/models/rnn.py for
    the Week 1 exercise skeleton (same interface, left unimplemented)."""

    @abstractmethod
    def forward(self, input_ids, **kwargs) -> HiddenStateOutput:
        """Run a forward pass. Must return hidden states for every layer
        (needed by Experiment 2.1's probing and Experiment 2.6's ablation)
        and logits (needed by every eval in Experiment 2.2)."""
        raise NotImplementedError

    @abstractmethod
    def get_attention_or_state(self, input_ids, **kwargs) -> Any:
        """Architecture-specific 'what is this layer looking at / carrying'
        signal, used by Experiment 2.1 Step 3 and Experiment 2.5 Step 3:
          - Attention-based (Transformer, GPT2, Hymba's attn branch):
            return attention weight tensors, shape (layer, head, seq, seq).
          - State-space (Mamba, Hymba's SSM branch): return the selective-
            scan Delta/B/C tensors (or whatever your implementation exposes)
            -- there are no attention weights, don't fake them.
          - Gated recurrent (sLSTM): return gate activations over time.
          - Plain recurrent (RNN): return hidden-state trajectories; there
            is no richer signal than that, which is the point of RNN being
            the lower-bound control (protocol document, Section 6, H6).
        Callers must not assume a fixed shape here -- branch on architecture
        type, or better, on what this method actually returns.
        """
        raise NotImplementedError

    @abstractmethod
    def num_params(self, *, non_embedding: bool = True) -> int:
        """Parameter count, used to verify the +-5% match required by
        Section 6, Step 1."""
        raise NotImplementedError

    @abstractmethod
    def save_checkpoint(self, path: str | Path, *, step: int, extra: dict | None = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def load_checkpoint(self, path: str | Path) -> dict:
        """Load weights in place; return the metadata dict saved alongside
        (step count etc.) so the trainer can resume correctly."""
        raise NotImplementedError
