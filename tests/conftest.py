"""Shared fixtures. Tests marked `data` skip automatically when raw datasets are absent."""
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw"


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO


@pytest.fixture(scope="session")
def raw_root() -> Path:
    return RAW


def requires_raw(*relpaths: str):
    """Decorator: skip a test unless every given data/raw/ path exists."""
    missing = [p for p in relpaths if not (RAW / p).exists()]
    return pytest.mark.skipif(bool(missing), reason=f"missing raw data: {missing}")
