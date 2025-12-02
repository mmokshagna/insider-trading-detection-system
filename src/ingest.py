"""Data ingestion utilities for insider trading datasets."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import pandas as pd


def _load_csv_files(paths: Iterable[Path]) -> List[pd.DataFrame]:
    frames: List[pd.DataFrame] = []
    for csv_path in paths:
        try:
            frames.append(pd.read_csv(csv_path))
        except FileNotFoundError:
            continue
    return frames


def load_insider_csvs(data_dir: str | Path = "data/processed") -> pd.DataFrame:
    """Load and concatenate insider trading CSV files from a directory."""

    directory = Path(data_dir)
    csv_files = sorted(directory.glob("*.csv"))
    frames = _load_csv_files(csv_files)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
