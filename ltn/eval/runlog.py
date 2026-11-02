"""Run logging: every experiment cell lands in results.jsonl (master plan §2.6).

Guarantee: "every table auto-generated from logged runs — no hand-copied
numbers anywhere". `RunLogger` writes one directory per (exp, model, seed)
containing config.yaml (config + git sha + start time), results.jsonl (one
JSON line per logged cell, exp/model/seed auto-stamped), and status.yaml
(ok/failed + traceback + duration). `read_results` is the ONLY input to
paper tables and figures.
"""
from __future__ import annotations

import json
import subprocess
import time
import traceback as _tb
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType

import pandas as pd
import yaml

__all__ = ["RunLogger", "read_results"]


def _git_sha() -> str | None:
    """Current repo HEAD, or None when git/repo is unavailable (e.g. deployed artifact)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


class RunLogger:
    """Context manager writing <root>/<exp>/<model>/<seed>/{config.yaml,results.jsonl,status.yaml}."""

    def __init__(self, exp: str, model: str, seed: int, config: dict, root: str | Path = "runs") -> None:
        self.exp = exp
        self.model = model
        self.seed = int(seed)
        self.config = dict(config)
        self.dir = Path(root) / exp / model / str(self.seed)
        self._t0: float | None = None

    def __enter__(self) -> "RunLogger":
        self.dir.mkdir(parents=True, exist_ok=True)
        self._t0 = time.monotonic()
        header = {
            "exp": self.exp,
            "model": self.model,
            "seed": self.seed,
            "git_sha": _git_sha(),
            "start_time": datetime.now(timezone.utc).isoformat(),
            "config": self.config,
        }
        with open(self.dir / "config.yaml", "w") as f:
            yaml.safe_dump(header, f, sort_keys=False)
        # a rerun of the same (exp, model, seed) replaces the run — stale rows
        # from a previous attempt must not double-count in paper tables
        (self.dir / "results.jsonl").write_text("")
        return self

    def log(self, **cell) -> None:
        """Append one result cell as a JSON line; exp/model/seed are stamped automatically."""
        rec = {"exp": self.exp, "model": self.model, "seed": self.seed, **cell}
        with open(self.dir / "results.jsonl", "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    def log_params(self, n_params: int) -> None:
        """Record model parameter count into config.yaml (E2 equal-param accounting)."""
        cfg_path = self.dir / "config.yaml"
        header = yaml.safe_load(cfg_path.read_text())
        header["n_params"] = int(n_params)
        with open(cfg_path, "w") as f:
            yaml.safe_dump(header, f, sort_keys=False)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        status: dict = {
            "status": "ok" if exc_type is None else "failed",
            "end_time": datetime.now(timezone.utc).isoformat(),
            "duration_s": None if self._t0 is None else round(time.monotonic() - self._t0, 3),
        }
        if exc_type is not None:
            status["traceback"] = "".join(_tb.format_exception(exc_type, exc, tb))
        with open(self.dir / "status.yaml", "w") as f:
            yaml.safe_dump(status, f, sort_keys=False)
        return False  # never suppress the exception


def read_results(root: str | Path, exp: str | None = None, model: str | None = None) -> pd.DataFrame:
    """Concatenate all results.jsonl under root (optionally filtered by exp/model)."""
    root = Path(root)
    rows: list[dict] = []
    for f in sorted(root.glob("*/*/*/results.jsonl")):
        run_exp, run_model, _seed = f.relative_to(root).parts[:3]
        if exp is not None and run_exp != exp:
            continue
        if model is not None and run_model != model:
            continue
        with open(f) as fh:
            rows.extend(json.loads(line) for line in fh if line.strip())
    return pd.DataFrame(rows)
