import gzip
import math

import pytest

from rtd.data.metrics import (
    NotePair,
    m1_repetitiveness,
    m2_beat_regularity,
    m3_dependency_distance,
    find_pairs_music21_heuristic,
)


def test_m1_repetitive_sequence_scores_higher_than_random():
    repetitive = [1, 2, 3, 4] * 200
    import random
    random.seed(0)
    rand_seq = [random.randint(0, 1000) for _ in range(800)]

    m1_rep = m1_repetitiveness(repetitive)
    m1_rand = m1_repetitiveness(rand_seq)

    assert 0.0 <= m1_rep <= 1.0
    assert 0.0 <= m1_rand <= 1.0
    assert m1_rep > m1_rand


def test_m1_empty_raises():
    with pytest.raises(ValueError):
        m1_repetitiveness([])


def test_m2_perfectly_regular_grid_is_near_one():
    onsets = [i * 0.5 for i in range(20)]  # perfectly even
    m2 = m2_beat_regularity(onsets)
    assert m2 == pytest.approx(1.0, abs=1e-6)


def test_m2_irregular_onsets_score_lower():
    regular = [i * 0.5 for i in range(20)]
    irregular = [0, 0.1, 0.4, 0.9, 1.0, 2.3, 2.35, 5.0, 5.01, 9.0]

    assert m2_beat_regularity(irregular) < m2_beat_regularity(regular)


def test_m2_requires_at_least_three_onsets():
    with pytest.raises(ValueError):
        m2_beat_regularity([0.0, 1.0])


def test_m3_mean_absolute_distance():
    pairs = [NotePair(0, 10), NotePair(5, 5), NotePair(2, 8)]
    # distances: 10, 0, 6 -> mean 16/3
    assert m3_dependency_distance(pairs) == pytest.approx(16 / 3)


def test_m3_empty_raises():
    with pytest.raises(ValueError):
        m3_dependency_distance([])


def test_find_pairs_heuristic_links_repeated_pitch_classes():
    pitches = ["C", "D", "E", "C", "F", "D"]
    pairs = find_pairs_music21_heuristic(pitches, window=10)
    positions = {(p.pos_i, p.pos_j) for p in pairs}
    assert (0, 3) in positions  # C at 0 links to C at 3
    assert (1, 5) in positions  # D at 1 links to D at 5
