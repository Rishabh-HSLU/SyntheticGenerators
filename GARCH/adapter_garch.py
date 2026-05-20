"""
adapter_garch.py
================
GARCH baseline adapter for the synthetic generator benchmark.

Fits a GJR-GARCH(1,1) model with Student-t innovations per ticker,
samples M synthetic windows of length window_len, inverse-normalizes
them, and saves the result as garch_synthetic.npy.

GJR-GARCH(1,1) is chosen over plain GARCH(1,1) because it captures
the leverage effect (asymmetric volatility response to positive vs
negative shocks) — one of the six fidelity buckets. Plain GARCH is
symmetric and would score zero on Bucket 4 by design.

Pipeline
--------
1. Load train_normalized.npy  (N_windows, window_len, 1)
2. Load norm_stats.json        per-ticker {mean, std}
3. Load ticker_split.json      to identify training tickers
4. For each training ticker:
      a. Reconstruct the full return series from all its windows
      b. Inverse-normalize (multiply by std, add mean)
      c. Fit GJR-GARCH(1,1) + Student-t
      d. Sample ceil(M / N_tickers) synthetic paths of length window_len
5. Pool synthetic paths across tickers, subsample to exactly M
6. Inverse-normalize is already done in step 4b — output is in
   deseasonalized-but-not-normalized units, matching eval_deseasonalized.npy
7. Save garch_synthetic.npy of shape (M, window_len, 1)

Usage
-----
    from adapter_garch import run_garch_pipeline

    run_garch_pipeline(
        output_dir  = "SyntheticGenerators/StylizedFactsAlignmentGAN/output_data",
        M           = 200,
        window_len  = 2520,
        output_path = "output_data/garch_synthetic.npy",
    )
"""

from __future__ import annotations

import json
import sys
import warnings
from math import ceil
from pathlib import Path

import numpy as np
from arch import arch_model

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from data_pipeline.canonical import BENCHMARK_N_PATHS, save_benchmark_corpus

# Suppress arch library warnings about non-stationary parameters during simulation.
# These are expected — some tickers have α+β near 1 due to COVID-period extremes,
# and the library safely falls back to intercept initialization.
warnings.filterwarnings("ignore", category=UserWarning, module="arch")
warnings.filterwarnings("ignore", message="Parameters are not consistent.*")


# ---------------------------------------------------------------------------
# Core: fit one GJR-GARCH(1,1) and sample paths
# ---------------------------------------------------------------------------

def fit_gjr_garch(returns: np.ndarray) -> object:
    """
    Fit GJR-GARCH(1,1) with Student-t innovations to a 1-D return series.

    Parameters
    ----------
    returns : 1-D array of deseasonalized log returns (not normalized)

    Returns
    -------
    fitted arch ModelResult object
    """
    model = arch_model(
        returns,
        mean    = 'Zero',    # financial returns have near-zero mean at 1-min
        vol     = 'GARCH',
        p       = 1,
        o       = 1,         # o=1 gives the asymmetric GJR term
        q       = 1,
        dist    = 'studentst',
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.fit(disp='off', show_warning=False)

    return result


def sample_gjr_garch(
    result:     object,
    n_paths:    int,
    window_len: int,
    seed:       int | None = None,
    max_std:    float | None = None,
    max_tries:  int  = 20,
) -> np.ndarray:
    """
    Sample synthetic paths from a fitted GJR-GARCH(1,1) model with
    rejection sampling for pathological draws.

    Parameters
    ----------
    result     : fitted arch ModelResult
    n_paths    : number of independent paths to sample
    window_len : length of each path in minutes
    seed       : random seed for reproducibility
    max_std    : reject any path with std exceeding this. If None,
                 no rejection (raw output).
    max_tries  : max attempts per path before giving up

    Returns
    -------
    paths : np.ndarray of shape (n_paths, window_len)
    """
    paths = []
    seed_offset = 0

    for i in range(n_paths):
        for attempt in range(max_tries):
            if seed is not None:
                np.random.seed(seed + i * max_tries + seed_offset)
            sim = result.model.simulate(
                result.params,
                nobs = window_len,
            )
            path = sim['data'].values

            if max_std is None or path.std() <= max_std:
                paths.append(path)
                break

            seed_offset += 1
        else:
            # All attempts produced pathological paths — append the last one
            # anyway and let downstream filtering handle it
            paths.append(path)

    return np.stack(paths)   # (n_paths, window_len), in original return units


# ---------------------------------------------------------------------------
# Reconstruct per-ticker series from windowed tensor
# ---------------------------------------------------------------------------

def reconstruct_ticker_series(
    train_arr:      np.ndarray,
    norm_stats:     dict,
    ticker_labels:  np.ndarray,
) -> dict[str, np.ndarray]:
    """Concatenate each ticker's windows and inverse z-score to deseasonalized units."""
    ticker_series: dict[str, np.ndarray] = {}
    for ticker in sorted(norm_stats):
        mask = ticker_labels == ticker
        if not mask.any():
            continue
        mean = norm_stats[ticker]["mean"]
        std = norm_stats[ticker]["std"]
        chunks = [train_arr[i, :, 0] * std + mean for i in np.flatnonzero(mask)]
        ticker_series[ticker] = np.concatenate(chunks)
    return ticker_series


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_garch_pipeline(
    output_dir:  str | Path,
    M:           int  = BENCHMARK_N_PATHS,
    window_len:  int  = 2520,
    output_path: str | Path | None = None,
    seed:        int  = 42,
) -> np.ndarray:
    """
    Full GARCH baseline pipeline.

    Parameters
    ----------
    output_dir  : directory containing train_normalized.npy,
                  norm_stats.json, ticker_split.json
    M           : number of synthetic windows to produce
    window_len  : length of each window in minutes (must match training)
    output_path : where to save garch_synthetic.npy
    seed        : global random seed

    Returns
    -------
    synthetic : np.ndarray of shape (M, window_len, 1)
                deseasonalized log returns, not normalized
    """
    output_dir = Path(output_dir)
    if output_path is None:
        output_path = output_dir / "garch_synthetic.npy"
    output_path = Path(output_path)

    # --- Load canonical outputs from data_prep ---
    train_arr = np.load(output_dir / "train_normalized.npy")
    print(f"Loaded train_normalized.npy : {train_arr.shape}")

    with open(output_dir / "norm_stats.json") as f:
        norm_stats = json.load(f)

    labels_path = output_dir / "train_ticker_labels.npy"
    if not labels_path.exists():
        raise FileNotFoundError(
            f"Missing {labels_path.name}; re-run data_prep.build_dataset"
        )
    ticker_labels = np.load(labels_path, allow_pickle=True)

    print(f"Training tickers with norm stats: {len(norm_stats)}")

    print("Reconstructing per-ticker return series...")
    ticker_series = reconstruct_ticker_series(train_arr, norm_stats, ticker_labels)
    print(f"Reconstructed {len(ticker_series)} ticker series")

    # --- Fit GJR-GARCH per ticker and sample ---
    paths_per_ticker = ceil(M / max(len(ticker_series), 1)) + 1
    all_paths = []
    failed    = 0

    print(f"Fitting GJR-GARCH(1,1) per ticker, sampling "
          f"{paths_per_ticker} paths each...")

    for i, (ticker, series) in enumerate(ticker_series.items()):
        if len(series) < window_len:
            failed += 1
            continue

        try:
            result = fit_gjr_garch(series)
            paths  = sample_gjr_garch(
                result,
                n_paths    = paths_per_ticker,
                window_len = window_len,
                seed       = seed + i,
                max_std    = 3.0 * series.std(),   # reject pathological draws
            )
            all_paths.append(paths)

        except Exception as e:
            failed += 1
            if failed <= 5:
                print(f"  [{ticker}] fit failed: {e}")
            continue

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(ticker_series)} tickers "
                  f"({failed} failed)")

    print(f"Fitting complete. Failed: {failed}/{len(ticker_series)}")

    if not all_paths:
        raise RuntimeError("No paths generated — all ticker fits failed.")

    # --- Pool and subsample to exactly M ---
    pooled = np.concatenate(all_paths, axis=0)   # (N_total, window_len)
    print(f"Total paths generated: {pooled.shape[0]}, subsampling to {M}")

    rng     = np.random.default_rng(seed)
    indices = rng.choice(len(pooled), size=M, replace=len(pooled) < M)
    synthetic = pooled[indices]                   # (M, window_len)

    print(f"\nExporting {M} paths to benchmark corpus...")
    synthetic = save_benchmark_corpus(synthetic, output_path, output_dir)
    print(f"Saved {output_path}  mean={synthetic.mean():.6f}  std={synthetic.std():.6f}")
    return np.load(output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fit GJR-GARCH(1,1) baseline and generate synthetic paths."
    )
    parser.add_argument("output_dir",  type=str)
    parser.add_argument("--M",         type=int, default=200)
    parser.add_argument("--window_len",type=int, default=2520)
    parser.add_argument("--output",    type=str, default=None)
    parser.add_argument("--seed",      type=int, default=42)
    args = parser.parse_args()

    run_garch_pipeline(
        output_dir  = args.output_dir,
        M           = args.M,
        window_len  = args.window_len,
        output_path = args.output,
        seed        = args.seed,
    )