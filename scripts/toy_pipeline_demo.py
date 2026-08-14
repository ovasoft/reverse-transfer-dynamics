#!/usr/bin/env python3
"""
End-to-end demo of the non-training pipeline on the toy corpus:
  data_samples/{abc_toy,midi_toy} -> ShardManifest -> tokenize -> M1/M2/M3

Run: PYTHONPATH=src python3 scripts/toy_pipeline_demo.py

This is also the backbone of the Week 1 exercise (see WEEK1_EXERCISE.md) --
if this script runs cleanly for you, your environment is set up correctly.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rtd.data.metrics import m1_repetitiveness
from rtd.data.sharding import ShardManifest
from rtd.tokenize.abc_tokenizer import ABCTokenizer, split_abc
from rtd.tokenize.remi_tokenizer import RemiTokenizerWrapper


def main() -> None:
    abc_dir = REPO_ROOT / "data_samples" / "abc_toy"
    midi_dir = REPO_ROOT / "data_samples" / "midi_toy"

    manifest = ShardManifest()
    for f in sorted(abc_dir.glob("*.abc")):
        manifest.add_file(f, format="ABC", genre="Folk")
    for f in sorted(midi_dir.glob("*.mid")):
        manifest.add_file(f, format="MIDI", genre="Synthetic")

    print(f"Shards registered: {len(manifest)}")

    # --- ABC shard: tokenize + M1 ---
    abc_shard = manifest.get("ABC:Folk")
    abc_texts = [Path(p).read_text() for p in abc_shard.file_paths]
    abc_tok = ABCTokenizer.build_from_corpus(abc_texts)
    all_abc_ids: list[int] = []
    for text in abc_texts:
        all_abc_ids.extend(abc_tok.encode(text, add_bos_eos=False))
    abc_m1 = m1_repetitiveness(all_abc_ids)
    print(f"\n[ABC:Folk] {len(abc_shard.file_paths)} tunes, "
          f"{len(all_abc_ids)} tokens, vocab={abc_tok.vocab_size}, M1={abc_m1:.3f}")

    # --- MIDI shard: tokenize + M1, per-file so we can see the M1 spread ---
    remi = RemiTokenizerWrapper()
    midi_shard = manifest.get("MIDI:Synthetic")
    print(f"\n[MIDI:Synthetic] {len(midi_shard.file_paths)} files:")
    for path in midi_shard.file_paths:
        ids = remi.encode_file(path)
        m1 = m1_repetitiveness(ids)
        print(f"  {Path(path).name:24s} tokens={len(ids):4d}  M1={m1:.3f}")

    print(
        "\nDone. Note: at this toy length (~150 tokens), gzip's fixed "
        "per-stream overhead makes M1 noisy -- arpeggio.mid (the most "
        "literally-repeated pattern, same 4 pitches/duration cycled) "
        "should reliably score highest, but scale_motif.mid's up-down "
        "shape repeats structurally without repeating literal byte runs "
        "as often, so it can score close to or below irregular_walk.mid "
        "at this scale. That's expected here, not a bug -- M1 needs "
        "longer sequences (real tunes, not 3-bar toys) before its ranking "
        "becomes reliable. Don't be alarmed if your own run's numbers "
        "differ slightly from what's in WEEK1_EXERCISE.md; the ordering "
        "of arpeggio.mid as highest-M1 is the one thing worth relying on."
    )


if __name__ == "__main__":
    main()
