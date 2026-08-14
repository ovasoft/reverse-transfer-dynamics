"""
Replay mixing -- see protocol document, Section 1 Step 6 (Model A/B: "5%
of the original musical-domain data interleaved uniformly throughout") and
Section 5 Step 3 (composition sweep: "5% replay drawn from that same run's
Phase 1 shard mixture -- not from the general shard pool").

The key design point (from the conversation that produced Section 5, Step
3): which shards replay is drawn from must match whatever Phase 1 corpus
this run actually used. `ReplayMixer` takes that shard-specific iterator
as a constructor argument rather than reaching into a global pool, so it's
impossible to accidentally wire the wrong shards in.
"""
from __future__ import annotations

import random
from typing import Iterator, TypeVar

T = TypeVar("T")


class ReplayMixer:
    def __init__(
        self,
        primary_stream: Iterator[T],
        replay_stream: Iterator[T],
        *,
        replay_fraction: float = 0.05,
        seed: int = 0,
    ):
        if not 0.0 <= replay_fraction < 1.0:
            raise ValueError("replay_fraction must be in [0, 1)")
        self.primary_stream = primary_stream
        self.replay_stream = replay_stream
        self.replay_fraction = replay_fraction
        self._rng = random.Random(seed)

    def __iter__(self) -> Iterator[tuple[T, str]]:
        return self

    def __next__(self) -> tuple[T, str]:
        """Yields (item, source) where source is 'primary' or 'replay', so
        callers/logging can verify the achieved ratio matches
        `replay_fraction` empirically rather than trusting it blindly."""
        if self._rng.random() < self.replay_fraction:
            return next(self.replay_stream), "replay"
        return next(self.primary_stream), "primary"


def cycle(stream_factory):
    """Wrap a zero-arg factory that returns a fresh iterator, so a replay
    stream (usually much shorter than the primary text stream) can be
    exhausted and restarted transparently instead of raising
    StopIteration partway through Phase 2."""
    while True:
        yield from stream_factory()
