"""
Dataset acquisition -- see protocol document, Section 1 Step 1.

Sources confirmed via web search at repo creation time (Aug 2026):
  - Lakh MIDI:   https://colinraffel.com/projects/lmd/  (CC-BY 4.0; homepage
                 hosts lmd_full.tar.gz -- confirm the exact filename on the
                 page before scripting the download, project pages sometimes
                 rename files across versions)
  - TheSession:  https://github.com/adactio/TheSession-data  (community-
                 maintained dumps of thesession.org; tunes.json/tunes.csv
                 contain the ABC-notation `abc` field directly)
  - RefinedWeb:  HF dataset `tiiuae/falcon-refinedweb` (ODC-By 1.0)

EMelodyGen was NOT independently verified (paper-associated datasets move
around HF/GitHub across releases) -- that one is still a TODO_VERIFY.

Regardless of source, call `record_provenance()` after every pull: Section 1
Step 1's fixed-seed/fixed-shard-ordering requirement (reused across all six
architectures in Experiment 2.5) only holds if the underlying data doesn't
silently change under you between runs.
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def record_provenance(dest_dir: str | Path, *, source: str, notes: str = "") -> None:
    """Append a provenance record so it's always possible to answer
    "which exact version of this dataset did we train on"."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    record_path = dest_dir / "data_provenance.json"
    records = json.loads(record_path.read_text()) if record_path.exists() else []
    records.append(
        {
            "source": source,
            "notes": notes,
            "pulled_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    record_path.write_text(json.dumps(records, indent=2))


def download_lakh_midi(dest_dir: str | Path, *, dry_run: bool = True) -> None:
    """Lakh MIDI Dataset (LMD-full), CC-BY 4.0. Homepage:
    https://colinraffel.com/projects/lmd/ -- confirm the exact tarball
    filename on that page before running for real (large download, tens of
    GB uncompressed); `dry_run=True` (default) only prints the plan.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    url = "https://colinraffel.com/projects/lmd/lmd_full.tar.gz"  # confirm filename on homepage first
    cmd = ["wget", "-c", url, "-P", str(dest_dir)]
    if dry_run:
        print("[dry_run] would run:", " ".join(cmd))
        return
    subprocess.run(cmd, check=True)
    subprocess.run(["tar", "-xzf", str(dest_dir / "lmd_full.tar.gz"), "-C", str(dest_dir)], check=True)
    record_provenance(dest_dir, source="lakh_midi", notes=f"pulled from {url}")


def download_thesession(dest_dir: str | Path, *, dry_run: bool = True) -> None:
    """TheSession-data (adactio/TheSession-data on GitHub), CC BY-SA --
    community-maintained dumps of thesession.org. `tunes.json` has one row
    per tune setting with an `abc` field containing the ABC notation body;
    genre/tune-type is in the `type` field (reel, jig, waltz, ...) which is
    a reasonable first genre-tagging signal for Section 1 Step 1's sharding.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    repo = "https://github.com/adactio/TheSession-data.git"
    cmd = ["git", "clone", "--depth", "1", repo, str(dest_dir / "TheSession-data")]
    if dry_run:
        print("[dry_run] would run:", " ".join(cmd))
        print("[dry_run] then read", dest_dir / "TheSession-data" / "json" / "tunes.json")
        return
    subprocess.run(cmd, check=True)
    record_provenance(dest_dir, source="thesession_data", notes=f"cloned {repo}")


def download_emelodygen(dest_dir: str | Path) -> None:
    """EMelodyGen dataset/toolkit.
    TODO_VERIFY: confirm the current hosting location (paper repo / HF
    dataset) before implementing -- not verified for this repo."""
    raise NotImplementedError(
        "TODO_VERIFY the current EMelodyGen release location before implementing."
    )


def stream_refinedweb_sample(n_docs: int = 10_000):
    """RefinedWeb text sample via HF `datasets` streaming (no full local
    download). Requires the `datasets` package (pyproject.toml [train]
    extra). Dataset id `tiiuae/falcon-refinedweb` confirmed current as of
    repo creation; it's public (ODC-By 1.0), no gated access needed.
    """
    from datasets import load_dataset  # local import: heavy optional dep

    ds = load_dataset("tiiuae/falcon-refinedweb", split="train", streaming=True)
    for i, row in enumerate(ds):
        if i >= n_docs:
            break
        yield row["content"]
