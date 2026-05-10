"""Return-series loaders for GARCH baselines.

Mirrors the SFAG data API but returns a flat 1-D return array per ticker
(GARCH operates directly on the time series, not on fixed-length windows).
"""

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def _load_returns(csv_path: str) -> np.ndarray:
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["returns"] = np.log(df["close"] / df["close"].shift(1))
    df = df.dropna(subset=["returns"])
    return df["returns"].values.astype(np.float64)   # arch wants float64


def load_single_ticker(csv_path: str) -> np.ndarray:
    """Load log-returns from one CSV. Returns shape (N,)."""
    r = _load_returns(csv_path)
    print(f"Loaded {os.path.basename(csv_path):<20s}  N={len(r):,}")
    return r


def load_multi_ticker(
    data_dir: str,
    ticker_list: Optional[List[str]] = None,
    min_length: int = 1000,
) -> Dict[str, np.ndarray]:
    """Load log-returns for multiple tickers from a directory of CSVs.

    Returns dict {ticker_name: returns_array}. Skips tickers with fewer
    than ``min_length`` observations.
    """
    files = ticker_list if ticker_list else [
        f for f in os.listdir(data_dir)
        if f.endswith(".csv") and f != "download_log.csv"
    ]

    out: Dict[str, np.ndarray] = {}
    skipped = 0
    for fname in files:
        try:
            r = _load_returns(os.path.join(data_dir, fname))
            if len(r) < min_length:
                skipped += 1
                continue
            out[fname.replace(".csv", "")] = r
        except Exception:
            skipped += 1

    print(f"Tickers loaded : {len(out)}")
    print(f"Tickers skipped: {skipped}")
    return out


def make_windows(returns: np.ndarray, T: int = 2520) -> np.ndarray:
    """Reshape into non-overlapping windows for evaluation against SFAG.

    Returns shape (n_windows, T, 1) — same layout SFAG's AlignmentModule expects.
    """
    n = len(returns) // T
    if n == 0:
        raise ValueError(f"Series too short: len={len(returns)}, T={T}")
    trimmed = returns[: n * T]
    return trimmed.reshape(n, T, 1).astype(np.float32)
