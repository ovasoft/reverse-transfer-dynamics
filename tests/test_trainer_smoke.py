"""
End-to-end smoke test: tiny synthetic data through Phase 1 -> Phase 2 on
the reference TransformerAdapter, checking loss goes down and checkpoints
land where expected. This is intentionally the same shape of exercise as
WEEK1_EXERCISE.md's toy training run -- if this test is green, the Week 1
exercise will work for students.

Skipped entirely if torch/transformers aren't installed.
"""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from rtd.models.transformer import TransformerAdapter, TOY_CONFIG
from rtd.train.checkpoint import CheckpointManager
from rtd.train.trainer import TrainConfig, run_phase1, run_phase2

VOCAB_SIZE = 50


def _random_batches(n_batches, batch_size=4, seq_len=16, vocab_size=VOCAB_SIZE):
    for _ in range(n_batches):
        yield torch.randint(0, vocab_size, (batch_size, seq_len))


def test_phase1_then_phase2_smoke(tmp_path):
    model = TransformerAdapter(vocab_size=VOCAB_SIZE, config_overrides=TOY_CONFIG)
    ckpt_mgr = CheckpointManager(tmp_path / "checkpoints")

    phase1_cfg = TrainConfig(
        checkpoint_every_tokens=10**9,  # irrelevant for phase1, only total_tokens matters
        total_tokens=4 * 16 * 20,  # 20 batches worth
        batch_size=4,
        seq_len=16,
    )
    log1 = run_phase1(model, _random_batches(30), phase1_cfg, ckpt_mgr)
    assert log1.tokens_seen >= phase1_cfg.total_tokens
    assert len(log1.losses) > 0

    phase1_ckpts = ckpt_mgr.list_checkpoints(phase="phase1")
    assert len(phase1_ckpts) == 1
    ref = ckpt_mgr.phase1_reference()
    assert ref["tokens_seen"] == log1.tokens_seen

    phase2_cfg = TrainConfig(
        checkpoint_every_tokens=4 * 16 * 5,  # checkpoint every 5 batches
        total_tokens=4 * 16 * 20,
        batch_size=4,
        seq_len=16,
    )
    log2 = run_phase2(
        model,
        _random_batches(30),
        _random_batches(30),
        phase2_cfg,
        ckpt_mgr,
        replay_fraction=0.05,
    )
    assert log2.tokens_seen >= phase2_cfg.total_tokens

    phase2_ckpts = ckpt_mgr.list_checkpoints(phase="phase2")
    assert len(phase2_ckpts) >= 3  # 20 batches / 5-batch cadence, roughly


def test_loss_decreases_on_repetitive_toy_data(tmp_path):
    """Not a rigorous convergence test -- just checks the training loop is
    actually updating weights, by training on a trivially learnable
    constant-ish sequence and confirming loss trends down."""
    model = TransformerAdapter(vocab_size=VOCAB_SIZE, config_overrides=TOY_CONFIG)
    ckpt_mgr = CheckpointManager(tmp_path / "checkpoints")

    def easy_batches(n):
        fixed = torch.randint(0, VOCAB_SIZE, (4, 16))
        for _ in range(n):
            yield fixed.clone()

    cfg = TrainConfig(
        checkpoint_every_tokens=10**9,
        total_tokens=4 * 16 * 60,
        batch_size=4,
        seq_len=16,
        learning_rate=1e-3,
    )
    log = run_phase1(model, easy_batches(60), cfg, ckpt_mgr)

    early = sum(log.losses[:5]) / 5
    late = sum(log.losses[-5:]) / 5
    assert late < early
