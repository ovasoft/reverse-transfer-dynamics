"""
Composition sweep design -- see protocol document, Section 5 (Experiment
2.4), Step 1: sample ~40 target points from the M1 x M2 x M3 cube via
Latin Hypercube Sampling, split ~20 ABC / ~20 MIDI, then build a Phase 1
corpus per point by selecting/weighting shards to approximate that target.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats.qmc import LatinHypercube, scale

from rtd.data.sharding import Shard, ShardManifest


@dataclass
class CompositionTarget:
    format: str  # "ABC" or "MIDI"
    m1: float
    m2: float
    m3: float


def sample_composition_targets(
    *,
    n_per_format: int = 20,
    formats: tuple[str, ...] = ("ABC", "MIDI"),
    m1_range: tuple[float, float] = (0.0, 1.0),
    m2_range: tuple[float, float] = (0.0, 1.0),
    m3_range: tuple[float, float] = (1.0, 200.0),
    seed: int = 0,
) -> list[CompositionTarget]:
    """Latin Hypercube sample over (M1, M2, M3), independently per format so
    Format stays a balanced factor in the regression (Section 5, Step 1).
    Ranges default to M1/M2's natural [0,1] bounds and a generic M3 token-
    distance range -- tighten these to your corpus's observed min/max
    before running the real sweep, since LHS coverage is only as good as
    the bounds it's given.
    """
    targets: list[CompositionTarget] = []
    for fi, fmt in enumerate(formats):
        sampler = LatinHypercube(d=3, seed=seed + fi)  # distinct seed per format, still reproducible
        unit_cube = sampler.random(n=n_per_format)
        scaled = scale(
            unit_cube,
            l_bounds=[m1_range[0], m2_range[0], m3_range[0]],
            u_bounds=[m1_range[1], m2_range[1], m3_range[1]],
        )
        for row in scaled:
            targets.append(CompositionTarget(format=fmt, m1=row[0], m2=row[1], m3=row[2]))
    return targets


def build_corpus_for_target(
    manifest: ShardManifest,
    target: CompositionTarget,
    *,
    max_shards: int = 10,
) -> list[Shard]:
    """Greedily select shards (of the target's format, any genre -- genre is
    deliberately left to vary naturally per the design decision in Section
    5, Step 1) that together best approximate the target (M1, M2, M3)
    point, by nearest-neighbor distance in metric space.

    This is a starting heuristic (nearest-shard selection), not a weighted-
    mixture optimizer -- a natural improvement is to weight/blend multiple
    shards' token streams so the *combined* corpus's measured M1/M2/M3 (not
    just the nearest single shard's) matches the target more closely.
    """
    candidates = [
        s
        for s in manifest.shards(format=target.format)
        if s.m1_repetitiveness is not None
        and s.m2_beat_regularity is not None
        and s.m3_dependency_distance is not None
    ]
    if not candidates:
        raise ValueError(f"No metric-annotated shards available for format={target.format}")

    def dist(s: Shard) -> float:
        return (
            (s.m1_repetitiveness - target.m1) ** 2
            + (s.m2_beat_regularity - target.m2) ** 2
            + ((s.m3_dependency_distance - target.m3) / 100.0) ** 2  # rough scale match to M1/M2
        ) ** 0.5

    ranked = sorted(candidates, key=dist)
    return ranked[:max_shards]


def achieved_metrics(shards: list[Shard]) -> tuple[float, float, float]:
    """Unweighted mean M1/M2/M3 across the selected shards -- record this
    as the *achieved* composition point (Section 5, Step 1: "record the
    achieved M1, M2, M3, not just the target"), since it will differ from
    the LHS target whenever the corpus doesn't have a shard sitting exactly
    on that point.
    """
    m1s = [s.m1_repetitiveness for s in shards]
    m2s = [s.m2_beat_regularity for s in shards]
    m3s = [s.m3_dependency_distance for s in shards]
    return (float(np.mean(m1s)), float(np.mean(m2s)), float(np.mean(m3s)))
