"""
Checkpoint manager -- see protocol document, Section 1, Step 6: "At each
checkpoint, export a frozen copy for evaluation. Do not continue training
the exported copy."
"""
from __future__ import annotations

import json
from pathlib import Path


class CheckpointManager:
    """Tracks checkpoints for one training run under `root_dir`, named by
    tokens-seen so Section 1's "every 0.8B tokens" cadence (or whatever
    interval your config uses) is unambiguous from the directory name
    alone -- don't rename these by hand, other tooling (eval scripts,
    Experiment 2.6's control-model comparisons) expects this format."""

    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root_dir / "checkpoints_index.json"
        self._index: list[dict] = (
            json.loads(self.index_path.read_text()) if self.index_path.exists() else []
        )

    def checkpoint_dir(self, tokens_seen: int) -> Path:
        return self.root_dir / f"tokens_{tokens_seen:012d}"

    def record(self, model, *, tokens_seen: int, phase: str, extra: dict | None = None) -> Path:
        """Save `model` (any SequenceModelAdapter) and register it in the
        index. `phase` should be "phase1" or "phase2" so downstream tooling
        can tell Section 1's Phase-1 reference checkpoint (used by
        Experiments 2.1 and 2.6) apart from Phase-2 checkpoints."""
        ckpt_dir = self.checkpoint_dir(tokens_seen)
        model.save_checkpoint(ckpt_dir, step=tokens_seen, extra=extra)
        entry = {"tokens_seen": tokens_seen, "phase": phase, "path": str(ckpt_dir), **(extra or {})}
        self._index.append(entry)
        self.index_path.write_text(json.dumps(self._index, indent=2))
        return ckpt_dir

    def list_checkpoints(self, *, phase: str | None = None) -> list[dict]:
        entries = sorted(self._index, key=lambda e: e["tokens_seen"])
        if phase is not None:
            entries = [e for e in entries if e["phase"] == phase]
        return entries

    def phase1_reference(self) -> dict:
        """The single Phase 1 end-state checkpoint (Section 1, Step 4:
        "this is the Phase 1 reference state used by Experiments 2.1 and
        2.6"). Raises if there isn't exactly one -- Phase 1 shouldn't be
        checkpointed more than once per Section 1's design (no
        intermediate Phase 1 checkpoints are used anywhere in the
        protocol)."""
        phase1 = self.list_checkpoints(phase="phase1")
        if len(phase1) != 1:
            raise ValueError(
                f"Expected exactly one phase1 checkpoint, found {len(phase1)}. "
                "Section 1's design only checkpoints once at the end of Phase 1."
            )
        return phase1[0]
