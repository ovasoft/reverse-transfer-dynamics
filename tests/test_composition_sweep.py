import pytest

from rtd.composition.sweep import (
    CompositionTarget,
    sample_composition_targets,
    build_corpus_for_target,
    achieved_metrics,
)
from rtd.data.sharding import ShardManifest


def test_sample_composition_targets_shape_and_bounds():
    targets = sample_composition_targets(n_per_format=15, seed=42)
    assert len(targets) == 30  # 15 ABC + 15 MIDI
    formats = {t.format for t in targets}
    assert formats == {"ABC", "MIDI"}
    for t in targets:
        assert 0.0 <= t.m1 <= 1.0
        assert 0.0 <= t.m2 <= 1.0
        assert 1.0 <= t.m3 <= 200.0


def test_sample_composition_targets_reproducible_with_seed():
    a = sample_composition_targets(n_per_format=10, seed=7)
    b = sample_composition_targets(n_per_format=10, seed=7)
    assert [(t.format, t.m1, t.m2, t.m3) for t in a] == [(t.format, t.m1, t.m2, t.m3) for t in b]


def _manifest_with_metrics(tmp_path):
    m = ShardManifest()
    specs = [
        ("Classical", 0.2, 0.8, 50.0),
        ("Jazz", 0.3, 0.2, 90.0),
        ("Electronic", 0.9, 0.9, 10.0),
        ("Folk", 0.5, 0.6, 30.0),
    ]
    for genre, m1, m2, m3 in specs:
        f = tmp_path / f"{genre}.abc"
        f.write_text(genre)
        m.add_file(f, format="ABC", genre=genre)
        sid = f"ABC:{genre}"
        m.set_metrics(sid, m1=m1, m2=m2, m3=m3)
    return m


def test_build_corpus_for_target_picks_nearest_shards(tmp_path):
    manifest = _manifest_with_metrics(tmp_path)
    target = CompositionTarget(format="ABC", m1=0.85, m2=0.85, m3=15.0)  # close to Electronic
    picked = build_corpus_for_target(manifest, target, max_shards=1)
    assert picked[0].genre == "Electronic"


def test_achieved_metrics_is_mean_of_selected_shards(tmp_path):
    manifest = _manifest_with_metrics(tmp_path)
    shards = manifest.shards(format="ABC")[:2]
    m1, m2, m3 = achieved_metrics(shards)
    expected_m1 = sum(s.m1_repetitiveness for s in shards) / 2
    assert m1 == pytest.approx(expected_m1)
