"""Build the full UTS parquet lake from downloaded raw datasets.

Usage: python -m ltn.data.build_lake [--raw data/raw] [--out data/uts] [--sources daqec,google_qec,...]

Each loader is independent; a failing source is reported and skipped so one bad
download never blocks the rest. Sim events are produced by `python -m ltn.sim.fleet`,
not here.
"""
from __future__ import annotations

import argparse
import traceback
from pathlib import Path

from ltn.data import uts


def build(raw_root: Path, out_root: Path, sources: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for src in sources:
        try:
            if src == "daqec":
                from ltn.data.loaders.daqec import load_daqec
                df = load_daqec(raw_root / "daqec")
            elif src == "daqec_v2":
                from ltn.data.loaders.daqec_v2 import load_daqec_v2
                df = load_daqec_v2(raw_root / "daqec_v2")
            elif src == "google_qec":
                from ltn.data.loaders.google import load_google
                df = load_google(raw_root / "google")
            elif src == "silicon":
                from ltn.data.loaders.silicon import load_silicon
                df = load_silicon(raw_root / "silicon")
            else:
                print(f"[build_lake] unknown source {src!r}; skipping")
                continue
            uts.write_uts(df, out_root)
            counts[src] = len(df)
            print(f"[build_lake] {src}: {len(df)} events")
        except Exception:
            counts[src] = -1
            print(f"[build_lake] FAILED {src}:\n{traceback.format_exc()}")
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/uts")
    ap.add_argument("--sources", default="daqec,daqec_v2,silicon,google_qec")
    args = ap.parse_args()
    counts = build(Path(args.raw), Path(args.out), args.sources.split(","))
    failed = [s for s, n in counts.items() if n < 0]
    if failed:
        raise SystemExit(f"failed sources: {failed}")


if __name__ == "__main__":
    main()
