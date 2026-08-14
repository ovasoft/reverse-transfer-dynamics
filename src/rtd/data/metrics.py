"""
Structural metrics M1, M2, M3 -- see protocol document, Section 1 Step 2
and Section 5 (Experiment 2.4).

These are computed per shard (a group of tunes/tracks sharing a genre tag)
and feed both the composition-sweep design (Experiment 2.4) and the
regression table.

All three functions are deliberately simple, dependency-light reference
implementations. They are starting points, not final answers -- in
particular m3_dependency_distance()'s note-pairing heuristic is a known
simplification (see its docstring) that the team should revisit.
"""
from __future__ import annotations

import gzip
import math
from dataclasses import dataclass
from typing import Iterable, Sequence


def m1_repetitiveness(tokens: Sequence[int] | bytes) -> float:
    """M1 = 1 - (compressed bytes / uncompressed bytes).

    High M1 -> highly redundant / motif-looping material (e.g. minimalist,
    electronic, strophic folk tunes). Low M1 -> little exploitable
    repetition at the token level.

    `tokens` can be a sequence of ints (token ids) or raw bytes; ints are
    packed to bytes first so compression operates on a fixed-width
    encoding rather than on Python's own tuple/list representation.
    """
    if isinstance(tokens, (bytes, bytearray)):
        raw = bytes(tokens)
    else:
        tokens = list(tokens)
        if not tokens:
            raise ValueError("m1_repetitiveness: empty token sequence")
        width = max(2, (max(tokens).bit_length() // 8) + 1)
        raw = b"".join(t.to_bytes(width, "little", signed=False) for t in tokens)

    if len(raw) == 0:
        raise ValueError("m1_repetitiveness: empty byte sequence")

    compressed = gzip.compress(raw, compresslevel=9)
    return 1.0 - (len(compressed) / len(raw))


def m2_beat_regularity(onset_times: Sequence[float]) -> float:
    """M2 = 1 / (sigma_dt + 1), sigma_dt = stdev of inter-onset time-deltas.

    `onset_times` should be sorted, monotonically increasing note-onset
    timestamps (seconds or ticks -- just be consistent within a corpus).
    Values near 1 -> a strict, unvarying rhythmic grid (techno, marches).
    Values near 0 -> fluid / syncopated / polyrhythmic material (jazz).
    """
    onset_times = sorted(onset_times)
    if len(onset_times) < 3:
        raise ValueError("m2_beat_regularity: need >= 3 onsets to get >= 2 deltas")

    deltas = [b - a for a, b in zip(onset_times, onset_times[1:])]
    mean = sum(deltas) / len(deltas)
    variance = sum((d - mean) ** 2 for d in deltas) / len(deltas)
    sigma = math.sqrt(variance)
    return 1.0 / (sigma + 1.0)


@dataclass
class NotePair:
    """A linked pair of symbolic positions -- e.g. a chord-resolution pair,
    or a thematic call/response repeat. `pos_i` / `pos_j` are token-index
    positions in the piece's tokenized sequence (not wall-clock time)."""

    pos_i: int
    pos_j: int


def m3_dependency_distance(pairs: Iterable[NotePair]) -> float:
    """M3 = mean absolute token-distance between linked note pairs.

    This function only does the averaging -- it deliberately does not
    decide what counts as a "linked pair". Producing `pairs` is the real
    music-theoretic work and belongs in a separate step:

      - `find_pairs_music21_heuristic()` below is a *placeholder*
        implementation using only pitch-class repetition as a stand-in for
        real chord-resolution / thematic-repeat detection. It will
        systematically underestimate M3 for pieces whose real dependency
        structure is harmonic rather than melodic-repetition-based. Treat
        it as a starting point to validate the pipeline end-to-end, and
        plan to replace or extend it (e.g. with music21's harmonic
        analysis, or a proper motif-detection pass) before trusting M3
        values for the regression in Experiment 2.4.
    """
    pairs = list(pairs)
    if not pairs:
        raise ValueError("m3_dependency_distance: no linked pairs given")
    return sum(abs(p.pos_i - p.pos_j) for p in pairs) / len(pairs)


def find_pairs_music21_heuristic(pitch_sequence: Sequence[str], window: int = 32) -> list[NotePair]:
    """Placeholder pair-finder: within a sliding `window`, link each note
    to the nearest later note sharing the same pitch class. This is a
    melodic-repetition proxy, NOT real harmonic/thematic dependency
    detection -- see m3_dependency_distance's docstring.

    `pitch_sequence` is a list of pitch-class strings (e.g. music21
    `note.pitch.name`, like "C", "F#") aligned 1:1 with token positions.
    """
    pairs: list[NotePair] = []
    n = len(pitch_sequence)
    for i in range(n):
        for j in range(i + 1, min(i + window, n)):
            if pitch_sequence[i] == pitch_sequence[j]:
                pairs.append(NotePair(i, j))
                break
    return pairs
