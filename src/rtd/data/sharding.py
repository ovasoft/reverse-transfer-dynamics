"""
Genre sharding and the shard manifest -- see protocol document, Section 1
Step 1 ("Acquire and shard the music corpus") and Step 2 ("Compute
per-shard structural metrics").

A "shard" here is a group of tunes/tracks that share (format, genre) --
e.g. all ABC/Classical tunes. The manifest records one row per shard with
its file list and, once computed, its M1/M2/M3 values. Experiment 2.4's
composition sweep (src/rtd/composition/sweep.py) samples directly from
this manifest.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Shard:
    shard_id: str
    format: str            # "ABC" or "MIDI"
    genre: str              # "Classical", "Jazz", "Folk", "Electronic", ...
    file_paths: list[str] = field(default_factory=list)
    m1_repetitiveness: float | None = None
    m2_beat_regularity: float | None = None
    m3_dependency_distance: float | None = None

    def add_file(self, path: str | Path) -> None:
        self.file_paths.append(str(path))


class ShardManifest:
    """In-memory + JSON-persisted registry of shards.

    Usage:
        manifest = ShardManifest()
        manifest.add_file("abc/tunes/reel_001.abc", format="ABC", genre="Folk")
        ...
        manifest.save("manifest.json")
        manifest2 = ShardManifest.load("manifest.json")
    """

    def __init__(self) -> None:
        self._shards: dict[str, Shard] = {}
        self._seen_hashes: set[str] = set()

    @staticmethod
    def _shard_id(format: str, genre: str) -> str:
        return f"{format}:{genre}"

    def add_file(self, path: str | Path, *, format: str, genre: str, dedupe: bool = True) -> bool:
        """Register a file under (format, genre). Returns False (and skips)
        if `dedupe` is True and a file with identical content hash was
        already added anywhere in the manifest -- this is the dedupe step
        called out in Section 1 Step 1 ("deduplicate at the tune/track
        level")."""
        path = Path(path)
        if dedupe:
            h = self._hash_file(path)
            if h in self._seen_hashes:
                return False
            self._seen_hashes.add(h)

        sid = self._shard_id(format, genre)
        if sid not in self._shards:
            self._shards[sid] = Shard(shard_id=sid, format=format, genre=genre)
        self._shards[sid].add_file(path)
        return True

    @staticmethod
    def _hash_file(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 16), b""):
                h.update(chunk)
        return h.hexdigest()

    def set_metrics(self, shard_id: str, *, m1: float, m2: float | None, m3: float | None) -> None:
        s = self._shards[shard_id]
        s.m1_repetitiveness = m1
        s.m2_beat_regularity = m2
        s.m3_dependency_distance = m3

    def shards(self, format: str | None = None, genre: str | None = None) -> list[Shard]:
        out = list(self._shards.values())
        if format is not None:
            out = [s for s in out if s.format == format]
        if genre is not None:
            out = [s for s in out if s.genre == genre]
        return out

    def get(self, shard_id: str) -> Shard:
        return self._shards[shard_id]

    def __len__(self) -> int:
        return len(self._shards)

    def save(self, path: str | Path) -> None:
        payload = {sid: asdict(s) for sid, s in self._shards.items()}
        Path(path).write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "ShardManifest":
        payload = json.loads(Path(path).read_text())
        m = cls()
        for sid, row in payload.items():
            m._shards[sid] = Shard(**row)
        return m
