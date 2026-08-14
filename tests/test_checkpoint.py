import json
from pathlib import Path

import pytest

from rtd.train.checkpoint import CheckpointManager


class _MockModel:
    """Stands in for a SequenceModelAdapter without needing torch --
    CheckpointManager only calls save_checkpoint/load_checkpoint, so a
    duck-typed mock is enough to test its bookkeeping logic in isolation."""

    def __init__(self):
        self.saved_at = []

    def save_checkpoint(self, path, *, step, extra=None):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "weights.txt").write_text(f"step={step}")
        self.saved_at.append(step)


def test_record_creates_checkpoint_dir_and_index_entry(tmp_path):
    mgr = CheckpointManager(tmp_path)
    model = _MockModel()

    ckpt_dir = mgr.record(model, tokens_seen=800_000_000, phase="phase2")

    assert ckpt_dir.exists()
    assert (ckpt_dir / "weights.txt").read_text() == "step=800000000"

    index = json.loads((tmp_path / "checkpoints_index.json").read_text())
    assert len(index) == 1
    assert index[0]["tokens_seen"] == 800_000_000
    assert index[0]["phase"] == "phase2"


def test_list_checkpoints_sorted_and_filterable(tmp_path):
    mgr = CheckpointManager(tmp_path)
    model = _MockModel()
    mgr.record(model, tokens_seen=0, phase="phase1")
    mgr.record(model, tokens_seen=1_600_000_000, phase="phase2")
    mgr.record(model, tokens_seen=800_000_000, phase="phase2")

    all_ckpts = mgr.list_checkpoints()
    assert [c["tokens_seen"] for c in all_ckpts] == [0, 800_000_000, 1_600_000_000]

    phase2_only = mgr.list_checkpoints(phase="phase2")
    assert len(phase2_only) == 2
    assert all(c["phase"] == "phase2" for c in phase2_only)


def test_phase1_reference_requires_exactly_one(tmp_path):
    mgr = CheckpointManager(tmp_path)
    model = _MockModel()

    with pytest.raises(ValueError):
        mgr.phase1_reference()  # none recorded yet

    mgr.record(model, tokens_seen=0, phase="phase1")
    ref = mgr.phase1_reference()
    assert ref["tokens_seen"] == 0

    mgr.record(model, tokens_seen=100, phase="phase1")  # a second one -- should now error
    with pytest.raises(ValueError):
        mgr.phase1_reference()


def test_index_persists_across_manager_instances(tmp_path):
    mgr1 = CheckpointManager(tmp_path)
    mgr1.record(_MockModel(), tokens_seen=42, phase="phase1")

    mgr2 = CheckpointManager(tmp_path)  # reload from disk
    assert len(mgr2.list_checkpoints()) == 1
    assert mgr2.list_checkpoints()[0]["tokens_seen"] == 42
