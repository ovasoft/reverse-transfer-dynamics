import json
from pathlib import Path

from rtd.data.sharding import ShardManifest


def test_add_file_and_dedupe(tmp_path: Path):
    f1 = tmp_path / "tune1.abc"
    f1.write_text("X:1\nT:Tune One\nK:C\nCDEF|")
    f2 = tmp_path / "tune1_copy.abc"
    f2.write_text("X:1\nT:Tune One\nK:C\nCDEF|")  # identical content -> dup
    f3 = tmp_path / "tune2.abc"
    f3.write_text("X:2\nT:Tune Two\nK:C\nGABC|")

    m = ShardManifest()
    assert m.add_file(f1, format="ABC", genre="Folk") is True
    assert m.add_file(f2, format="ABC", genre="Folk") is False  # deduped
    assert m.add_file(f3, format="ABC", genre="Folk") is True

    shards = m.shards(format="ABC", genre="Folk")
    assert len(shards) == 1
    assert len(shards[0].file_paths) == 2  # f1 and f3, not the dup


def test_separate_shards_per_genre(tmp_path: Path):
    m = ShardManifest()
    a = tmp_path / "a.abc"; a.write_text("aaa")
    b = tmp_path / "b.abc"; b.write_text("bbb")
    m.add_file(a, format="ABC", genre="Jazz")
    m.add_file(b, format="ABC", genre="Classical")
    assert len(m) == 2
    assert {s.genre for s in m.shards()} == {"Jazz", "Classical"}


def test_metrics_roundtrip_via_save_load(tmp_path: Path):
    m = ShardManifest()
    f = tmp_path / "x.abc"; f.write_text("xyz")
    m.add_file(f, format="ABC", genre="Electronic")
    sid = m.shards()[0].shard_id
    m.set_metrics(sid, m1=0.7, m2=0.5, m3=12.3)

    out = tmp_path / "manifest.json"
    m.save(out)
    m2 = ShardManifest.load(out)
    s = m2.get(sid)
    assert s.m1_repetitiveness == 0.7
    assert s.m3_dependency_distance == 12.3
