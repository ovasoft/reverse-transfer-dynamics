"""
Phase 1 / Phase 2 training loop -- see protocol document, Section 1,
Steps 4 and 6.

Deliberately minimal (plain PyTorch, no distributed/mixed-precision
framework) -- at the 20M-0.6B parameter scale this project uses (Section 1
and Section 5's compute-budget tables), a hand-rolled loop is enough, and
it's easier for the team to actually understand and debug than wiring in
DeepSpeed/FSDP for models this small. If a later architecture genuinely
needs more, that's a deliberate upgrade to make then, not a default now.

Requires torch (pyproject.toml [train] extra) -- not execution-tested in
the environment this repo was built in. Run
tests/test_trainer_smoke.py once torch is available.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterator

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from rtd.models.base import SequenceModelAdapter
from rtd.train.checkpoint import CheckpointManager
from rtd.train.replay import ReplayMixer


@dataclass
class TrainConfig:
    checkpoint_every_tokens: int
    total_tokens: int
    batch_size: int
    seq_len: int
    learning_rate: float = 3e-4
    grad_clip: float = 1.0
    device: str = "cpu"


@dataclass
class TrainLog:
    tokens_seen: int = 0
    losses: list[float] = field(default_factory=list)

    def record(self, tokens_this_step: int, loss: float) -> None:
        self.tokens_seen += tokens_this_step
        self.losses.append(loss)


def _lm_loss(logits, input_ids):
    """Standard next-token cross-entropy, shifted by one position."""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    return torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )


def run_phase1(
    model: SequenceModelAdapter,
    data_iterator: Iterator,  # yields (batch,) tensors of token ids, shape (batch, seq_len)
    config: TrainConfig,
    checkpoint_mgr: CheckpointManager,
) -> TrainLog:
    """Section 1, Step 4: train from scratch on the music-only stream to
    `config.total_tokens`, checkpoint exactly once at the end (Step 4's
    third bullet -- no intermediate Phase 1 checkpoints)."""
    if not _TORCH_AVAILABLE:
        raise ImportError("torch not installed. Install with: pip install 'rtd[train]'")

    optimizer = torch.optim.AdamW(model.model.parameters(), lr=config.learning_rate)
    log = TrainLog()

    for batch in data_iterator:
        batch = batch.to(config.device)
        out = model.forward(batch)
        loss = _lm_loss(out.logits, batch)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.model.parameters(), config.grad_clip)
        optimizer.step()

        tokens_this_step = batch.numel()
        log.record(tokens_this_step, loss.item())

        if log.tokens_seen >= config.total_tokens:
            break

    checkpoint_mgr.record(model, tokens_seen=log.tokens_seen, phase="phase1")
    return log


def run_phase2(
    model: SequenceModelAdapter,
    text_iterator: Iterator,
    replay_iterator: Iterator,
    config: TrainConfig,
    checkpoint_mgr: CheckpointManager,
    *,
    replay_fraction: float = 0.05,
    on_checkpoint: Callable[[int], None] | None = None,
) -> TrainLog:
    """Section 1, Step 6: text adaptation with replay mixed in at
    `replay_fraction` (5% by default, matching Model A/B's design; the
    composition sweep in Section 5 Step 3 uses the same default but a
    replay_iterator built from that run's own Phase 1 shards, not the
    general pool -- that distinction lives in what you pass in here, not
    in this function). Checkpoints every `config.checkpoint_every_tokens`.
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("torch not installed. Install with: pip install 'rtd[train]'")

    optimizer = torch.optim.AdamW(model.model.parameters(), lr=config.learning_rate)
    mixer = ReplayMixer(text_iterator, replay_iterator, replay_fraction=replay_fraction)
    log = TrainLog()
    next_checkpoint_at = config.checkpoint_every_tokens

    for batch, _source in mixer:
        batch = batch.to(config.device)
        out = model.forward(batch)
        loss = _lm_loss(out.logits, batch)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.model.parameters(), config.grad_clip)
        optimizer.step()

        tokens_this_step = batch.numel()
        log.record(tokens_this_step, loss.item())

        if log.tokens_seen >= next_checkpoint_at:
            checkpoint_mgr.record(model, tokens_seen=log.tokens_seen, phase="phase2")
            if on_checkpoint is not None:
                on_checkpoint(log.tokens_seen)
            next_checkpoint_at += config.checkpoint_every_tokens

        if log.tokens_seen >= config.total_tokens:
            break

    return log
