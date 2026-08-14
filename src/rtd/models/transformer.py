"""
Reference SequenceModelAdapter implementation, backed by Hugging Face's
GPT2LMHeadModel. Used for BOTH the "standard causal Transformer" (Model
A/B's original architecture) and the GPT2 entry in Experiment 2.5's
architecture spectrum (Section 6, Step 1) -- they're the same underlying
class with different configs, since the protocol's "Transformer" is itself
attention-only/decoder-only, same family as GPT2.

This is the "golden path" adapter: get every other architecture adapter
working by comparing its behavior against this one on the same toy data.

NOTE: this module requires `torch` and `transformers` (pyproject.toml
[train] extra). It could not be execution-tested in the environment this
starter repo was built in (no torch available there) -- read it carefully
and run `tests/test_transformer_adapter.py` (which is skipped if torch
isn't installed) as the first thing you do once your environment has
torch, to confirm it behaves as documented before building on top of it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from rtd.models.base import HiddenStateOutput, SequenceModelAdapter

try:
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# Section 1's compute-budget table: ~0.6B non-embedding params, matched to
# the Qwen3-0.6B / Qwen2.5-0.5B scale. This GPT2Config is a starting point
# to hit roughly that scale -- verify actual non-embedding param count with
# `num_params(non_embedding=True)` and adjust n_layer/n_embd if it drifts,
# rather than trusting these numbers blindly.
DEFAULT_0_6B_CONFIG = dict(
    n_layer=24,
    n_embd=1024,
    n_head=16,
    n_positions=2048,
    n_ctx=2048,
)

# Small config for fast unit tests / the Week 1 toy exercise -- NOT a
# research-scale model, just big enough to exercise the code paths.
TOY_CONFIG = dict(n_layer=2, n_embd=32, n_head=2, n_positions=128, n_ctx=128)


class TransformerAdapter(SequenceModelAdapter):
    def __init__(self, vocab_size: int, *, config_overrides: dict | None = None, device: str = "cpu"):
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "torch/transformers not installed. Install with: pip install 'rtd[train]'"
            )
        cfg_kwargs = {**DEFAULT_0_6B_CONFIG, **(config_overrides or {})}
        config = GPT2Config(vocab_size=vocab_size, **cfg_kwargs)
        self.model = GPT2LMHeadModel(config)
        self.device = device
        self.model.to(device)

    def forward(self, input_ids, attention_mask=None, **kwargs) -> HiddenStateOutput:
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=True,
            **kwargs,
        )
        return HiddenStateOutput(
            hidden_states=out.hidden_states,
            logits=out.logits,
            attention_or_state=out.attentions,  # tuple, one (batch, head, seq, seq) tensor per layer
        )

    def get_attention_or_state(self, input_ids, **kwargs) -> Any:
        return self.forward(input_ids, **kwargs).attention_or_state

    def num_params(self, *, non_embedding: bool = True) -> int:
        total = sum(p.numel() for p in self.model.parameters())
        if not non_embedding:
            return total
        embed_params = sum(p.numel() for p in self.model.transformer.wte.parameters())
        embed_params += sum(p.numel() for p in self.model.transformer.wpe.parameters())
        return total - embed_params

    def save_checkpoint(self, path: str | Path, *, step: int, extra: dict | None = None) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        torch.save({"step": step, **(extra or {})}, path / "trainer_state.pt")

    def load_checkpoint(self, path: str | Path) -> dict:
        path = Path(path)
        self.model = GPT2LMHeadModel.from_pretrained(path).to(self.device)
        state_path = path / "trainer_state.pt"
        return torch.load(state_path) if state_path.exists() else {}


def build_gpt2_reference(vocab_size: int, *, device: str = "cpu") -> TransformerAdapter:
    """Experiment 2.5's GPT2 entry: same class, explicitly using GPT2's own
    published small config instead of the project's custom 0.6B config, so
    results are checkable against published GPT2 behavior (Section 6,
    Step 1 table)."""
    gpt2_small = dict(n_layer=12, n_embd=768, n_head=12, n_positions=1024, n_ctx=1024)
    return TransformerAdapter(vocab_size, config_overrides=gpt2_small, device=device)
