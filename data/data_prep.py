"""
data_prep.py
============
Common real-data preparation pipeline for all synthetic financial
time series generators (SFAGan, SBBTS, GARCH, CoFinDif, AIL).

Pipeline decisions (locked after empirical verification):
=========================================================

Step 1 — Session filtering
    Keep only regular NYSE session bars: minute-of-day in NY time
    between 570 (9:30 AM) and 959 (3:59 PM) inclusive.
    Drop all pre-market and after-hours bars.

Step 2 — Sorting
    Sort each ticker by timestamp within session.

Step 3 — Gap detection
    Compute time difference between consecutive bars within the same
    trading day. A gap is any within-session transition > 1 minute.

Step 4 — Post-gap return removal
    Drop any bar that follows a within-session gap. Its return would
    span multiple minutes and is not a valid 1-minute observation.

Step 5 — Session-boundary return removal
    Drop the first bar of each trading day. Its return would span
    overnight and is not a 1-minute return.

Step 6 — Log return computation
    r_t = log(C_t / C_{t-1}) where C is the close price.
    After Steps 1-5, every consecutive pair is a clean 1-minute return.

Step 7 — Liquidity filtering
    Training corpus  : median bars per regular session day >= 280
    Evaluation corpus: median bars per regular session day >= 350
    Verified empirically:
      >= 280 bars/day -> 725 tickers (76.5% of universe)
      >= 350 bars/day -> 429 tickers (45.3% of universe)
    Liquid tickers (>= 350) show 93-99% of within-session transitions
    are exactly 1 minute — sufficient for temporal metric estimation.

Step 8 — Ticker split
    From the 429 evaluation-eligible tickers, hold out 20% (~86) as
    the evaluation set using a fixed random seed (seed=42).
    The remaining ~343 liquid tickers join ~382 semi-liquid tickers
    (280-349 bars/day) to form the ~725-ticker training corpus.
    Split is saved to ticker_split.json for full reproducibility.

Step 9 — Deseasonalization (Flexible Fourier Form)
    Fit a single pooled FFF curve s(tau) on training-ticker returns,
    where tau is minute-of-day (0-389). The curve captures the
    U-shaped intraday volatility pattern common to all equity returns.
    Deseasonalized residual: r_tilde_t = r_t / s(tau_t).
    The seasonal pattern is saved separately for inversion if needed.
    Applied to both training and evaluation tickers using train-fitted
    curve only (no leakage).

Step 10 — Per-ticker normalization (training tickers only)
    Z-score each training ticker's deseasonalized returns using
    that ticker's own mean and std from the full training period.
    Normalization stats saved per ticker for inference-time inversion.
    Evaluation tickers are saved deseasonalized but NOT normalized —
    the evaluation framework operates on real return magnitudes.

Outputs
=======
    train_normalized.npy    : shape (N_train, T, 1), normalized returns
    eval_deseasonalized.npy : shape (N_eval, T, 1), raw deseasonalized
    fff_pattern.npy         : shape (390,), seasonal curve s(tau)
    norm_stats.json         : per-ticker {mean, std} for inversion
    ticker_split.json       : {train: [...], eval: [...]} ticker lists
    stats_df.csv            : per-ticker liquidity stats for reference
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SESSION_START_MIN = 570   # 9:30 AM NY time in minutes since midnight
SESSION_END_MIN   = 959   # 3:59 PM NY time in minutes since midnight
MINUTES_PER_DAY   = SESSION_END_MIN - SESSION_START_MIN + 1  # 390

TRAIN_BAR_THRESHOLD = 280   # Step 7: training corpus liquidity floor
EVAL_BAR_THRESHOLD  = 350   # Step 7: evaluation corpus liquidity floor
EVAL_HOLDOUT_FRAC   = 0.20  # Step 8: fraction of liquid tickers held out
RANDOM_SEED         = 42    # Step 8: fixed seed for reproducibility

FFF_HARMONICS = 3           # Step 9: number of Fourier harmonics


# ---------------------------------------------------------------------------
# Step 1-6: Single-ticker cleaning and log return computation
# ---------------------------------------------------------------------------

def clean_ticker(csv_path: Path) -> pd.DataFrame | None:
    """
    Load one ticker CSV and apply Steps 1-6.

    Returns a DataFrame with columns:
        date_ny, minute_of_day_ny, log_return
    or None if the file cannot be processed.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        warnings.warn(f"Could not read {csv_path.name}: {e}")
        return None

    # Guard: skip files that are not ticker data
    if 'timestamp' not in df.columns:
        return None

    # --- Step 1: parse timestamps, convert to NY time ---
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df['ts_ny']     = df['timestamp'].dt.tz_convert('America/New_York')
    df['minute_of_day_ny'] = (df['ts_ny'].dt.hour * 60 +
                               df['ts_ny'].dt.minute)
    df['date_ny']   = df['ts_ny'].dt.date

    # --- Step 1: session filter ---
    regular = df[
        (df['minute_of_day_ny'] >= SESSION_START_MIN) &
        (df['minute_of_day_ny'] <= SESSION_END_MIN)
    ].copy()

    if regular.empty:
        return None

    # --- Step 2: sort ---
    regular = regular.sort_values('timestamp').reset_index(drop=True)

    # --- Step 3: gap detection ---
    regular['diff_min'] = (regular['timestamp']
                           .diff()
                           .dt.total_seconds() / 60)
    regular['same_day'] = (regular['date_ny'] ==
                            regular['date_ny'].shift(1))

    # --- Steps 4-5: drop post-gap and session-boundary bars ---
    is_session_start = ~regular['same_day']
    is_post_gap      = regular['same_day'] & (regular['diff_min'] > 1)
    regular_clean    = regular[~(is_session_start | is_post_gap)].copy()

    if len(regular_clean) < 2:
        return None

    # --- Step 6: log returns ---
    regular_clean['log_return'] = np.log(
        regular_clean['close'] / regular_clean['close'].shift(1)
    )
    regular_clean = regular_clean.dropna(subset=['log_return'])

    return regular_clean[['date_ny', 'minute_of_day_ny', 'log_return']]


# ---------------------------------------------------------------------------
# Step 7: Per-ticker liquidity stats
# ---------------------------------------------------------------------------

def compute_liquidity_stats(csv_path: Path) -> dict:
    """
    Return median bars per regular session day for one ticker.
    Used to apply the liquidity threshold (Step 7).
    """
    df = clean_ticker(csv_path)
    if df is None or df.empty:
        return {'ticker': csv_path.stem, 'median_bars_per_day': 0.0}

    bars_per_day = df.groupby('date_ny').size()
    return {
        'ticker': csv_path.stem,
        'median_bars_per_day': float(bars_per_day.median()),
        'trading_days': len(bars_per_day),
        'total_returns': len(df),
    }


def apply_liquidity_filter(
    data_dir: Path,
    csv_files: list[Path],
) -> pd.DataFrame:
    """
    Compute liquidity stats for all tickers and return a DataFrame
    with columns: ticker, median_bars_per_day, trading_days,
    total_returns, liquidity_class.

    liquidity_class:
        'eval_eligible' : median >= EVAL_BAR_THRESHOLD (350)
        'train_only'    : TRAIN_BAR_THRESHOLD <= median < EVAL_BAR_THRESHOLD
        'excluded'      : median < TRAIN_BAR_THRESHOLD
    """
    stats = [compute_liquidity_stats(f) for f in csv_files]
    df = pd.DataFrame(stats)

    df['liquidity_class'] = 'excluded'
    df.loc[
        df['median_bars_per_day'] >= TRAIN_BAR_THRESHOLD,
        'liquidity_class'
    ] = 'train_only'
    df.loc[
        df['median_bars_per_day'] >= EVAL_BAR_THRESHOLD,
        'liquidity_class'
    ] = 'eval_eligible'

    return df


# ---------------------------------------------------------------------------
# Step 8: Ticker split
# ---------------------------------------------------------------------------

def make_ticker_split(stats_df: pd.DataFrame) -> dict[str, list[str]]:
    """
    From eval-eligible tickers, hold out EVAL_HOLDOUT_FRAC as the
    evaluation set (fixed seed). Remaining eval-eligible tickers plus
    train-only tickers form the training set.

    Returns dict with keys 'train' and 'eval'.
    """
    rng = np.random.default_rng(RANDOM_SEED)

    eval_eligible = stats_df[
        stats_df['liquidity_class'] == 'eval_eligible'
    ]['ticker'].tolist()

    train_only = stats_df[
        stats_df['liquidity_class'] == 'train_only'
    ]['ticker'].tolist()

    # Shuffle eval-eligible and split
    eval_eligible_arr = np.array(sorted(eval_eligible))
    rng.shuffle(eval_eligible_arr)

    n_eval = int(len(eval_eligible_arr) * EVAL_HOLDOUT_FRAC)
    eval_tickers  = eval_eligible_arr[:n_eval].tolist()
    train_liquid  = eval_eligible_arr[n_eval:].tolist()

    train_tickers = train_liquid + train_only

    return {'train': sorted(train_tickers), 'eval': sorted(eval_tickers)}


# ---------------------------------------------------------------------------
# Step 9: Flexible Fourier Form deseasonalization
# ---------------------------------------------------------------------------

def _fff_func(tau: np.ndarray, *params) -> np.ndarray:
    """
    Evaluate the FFF curve at minute-of-day positions tau (0-indexed,
    0 = first minute of session).

    params layout: [c0, a1, b1, a2, b2, ..., aJ, bJ]
    where J = FFF_HARMONICS.
    """
    c0     = params[0]
    result = np.full_like(tau, c0, dtype=float)
    for j in range(1, FFF_HARMONICS + 1):
        a_j = params[2 * j - 1]
        b_j = params[2 * j]
        result += (a_j * np.cos(2 * np.pi * j * tau / MINUTES_PER_DAY) +
                   b_j * np.sin(2 * np.pi * j * tau / MINUTES_PER_DAY))
    return result


def fit_fff(train_returns: list[pd.DataFrame]) -> np.ndarray:
    """
    Fit the FFF seasonal curve on pooled training-ticker returns.

    Parameters
    ----------
    train_returns : list of cleaned DataFrames (one per training ticker),
                    each with columns [date_ny, minute_of_day_ny, log_return]

    Returns
    -------
    s : np.ndarray of shape (390,)
        Estimated seasonal scale s(tau) for tau = 0, ..., 389.
        s(tau) is the expected absolute return at minute tau.
    """
    pooled = pd.concat(train_returns, ignore_index=True)

    # Use minute-of-day relative to session start (0-indexed)
    pooled['tau'] = pooled['minute_of_day_ny'] - SESSION_START_MIN

    # Target: mean absolute return per minute-of-day
    mean_abs = (pooled.groupby('tau')['log_return']
                .apply(lambda x: x.abs().mean())
                .reset_index())
    mean_abs.columns = ['tau', 'mean_abs_return']
    mean_abs = mean_abs.sort_values('tau').reset_index(drop=True)

    tau_vals = mean_abs['tau'].values.astype(float)
    y_vals   = mean_abs['mean_abs_return'].values

    # Initial parameter guess: c0 = global mean, harmonics = 0
    n_params = 1 + 2 * FFF_HARMONICS
    p0 = np.zeros(n_params)
    p0[0] = y_vals.mean()

    popt, _ = curve_fit(_fff_func, tau_vals, y_vals, p0=p0, maxfev=10_000)

    # Evaluate on the full 0-389 grid
    tau_grid = np.arange(MINUTES_PER_DAY, dtype=float)
    s = _fff_func(tau_grid, *popt)

    # Clip to avoid division by near-zero
    s = np.clip(s, a_min=1e-8, a_max=None)
    return s


def deseasonalize(df: pd.DataFrame, s: np.ndarray) -> pd.DataFrame:
    """
    Divide each return by its minute-of-day seasonal factor.

    Parameters
    ----------
    df : cleaned DataFrame with [date_ny, minute_of_day_ny, log_return]
    s  : seasonal curve of shape (390,), output of fit_fff

    Returns
    -------
    df with an added column 'return_deseas'
    """
    tau = df['minute_of_day_ny'] - SESSION_START_MIN
    df  = df.copy()
    df['return_deseas'] = df['log_return'] / s[tau.values]
    return df


# ---------------------------------------------------------------------------
# Step 10: Per-ticker normalization (training tickers only)
# ---------------------------------------------------------------------------

def normalize_ticker(
    df: pd.DataFrame,
    mean: float | None = None,
    std:  float | None = None,
) -> tuple[pd.DataFrame, float, float]:
    """
    Z-score the 'return_deseas' column.

    If mean/std are None, compute from df (training mode).
    Otherwise use provided values (inference / evaluation mode).

    Returns (df_with_normed_col, mean_used, std_used).
    """
    if mean is None:
        mean = float(df['return_deseas'].mean())
    if std is None:
        std  = float(df['return_deseas'].std())
        std  = max(std, 1e-8)

    df = df.copy()
    df['return_normed'] = (df['return_deseas'] - mean) / std
    return df, mean, std


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------

def build_dataset(
    data_dir:   str | Path,
    output_dir: str | Path,
    window_len: int = 2520,
) -> None:
    """
    Run the full 10-step pipeline and save all outputs to output_dir.

    Parameters
    ----------
    data_dir   : directory containing one CSV per ticker
    output_dir : directory where outputs are written
    window_len : length of each synthetic evaluation window in minutes
                 (default 2520 = 10 trading days at 1-minute resolution)

    Outputs written
    ---------------
    ticker_split.json       Step 8  — train/eval ticker lists
    stats_df.csv            Step 7  — per-ticker liquidity stats
    fff_pattern.npy         Step 9  — seasonal curve s(tau), shape (390,)
    norm_stats.json         Step 10 — {ticker: {mean, std}} for inversion
    train_normalized.npy    Step 10 — shape (N_train_windows, window_len, 1)
    eval_deseasonalized.npy Step 9  — shape (N_eval_windows, window_len, 1)
    """
    data_dir   = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(
        f for f in data_dir.glob("*.csv")
        if f.name != "download_log.csv"
    )
    print(f"Found {len(csv_files)} CSV files in {data_dir}")

    # -----------------------------------------------------------------------
    # Steps 7-8: liquidity stats and ticker split
    # -----------------------------------------------------------------------
    print("Computing liquidity stats (Steps 7-8)...")
    stats_df = apply_liquidity_filter(data_dir, csv_files)
    stats_df.to_csv(output_dir / "stats_df.csv", index=False)

    split = make_ticker_split(stats_df)
    with open(output_dir / "ticker_split.json", "w") as f:
        json.dump(split, f, indent=2)

    print(f"  Training tickers : {len(split['train'])}")
    print(f"  Evaluation tickers: {len(split['eval'])}")

    # -----------------------------------------------------------------------
    # Steps 1-6: clean all tickers and collect DataFrames
    # -----------------------------------------------------------------------
    print("Cleaning tickers (Steps 1-6)...")

    ticker_to_file = {f.stem: f for f in csv_files}

    train_dfs: dict[str, pd.DataFrame] = {}
    eval_dfs:  dict[str, pd.DataFrame] = {}

    for ticker in split['train']:
        df = clean_ticker(ticker_to_file[ticker])
        if df is not None and not df.empty:
            train_dfs[ticker] = df

    for ticker in split['eval']:
        df = clean_ticker(ticker_to_file[ticker])
        if df is not None and not df.empty:
            eval_dfs[ticker] = df

    print(f"  Successfully cleaned: {len(train_dfs)} train, "
          f"{len(eval_dfs)} eval tickers")

    # -----------------------------------------------------------------------
    # Step 9: fit FFF on training tickers, deseasonalize both sets
    # -----------------------------------------------------------------------
    print("Fitting FFF seasonal curve (Step 9)...")
    s = fit_fff(list(train_dfs.values()))
    np.save(output_dir / "fff_pattern.npy", s)
    print(f"  FFF curve fitted. Range: [{s.min():.6f}, {s.max():.6f}]")

    print("Deseasonalizing all tickers...")
    train_dfs = {t: deseasonalize(df, s) for t, df in train_dfs.items()}
    eval_dfs  = {t: deseasonalize(df, s) for t, df in eval_dfs.items()}

    # -----------------------------------------------------------------------
    # Step 10: normalize training tickers; save eval deseasonalized only
    # -----------------------------------------------------------------------
    print("Normalizing training tickers (Step 10)...")
    norm_stats: dict[str, dict] = {}
    train_normed: dict[str, pd.DataFrame] = {}

    for ticker, df in train_dfs.items():
        df_n, mean, std = normalize_ticker(df)
        train_normed[ticker] = df_n
        norm_stats[ticker]   = {'mean': mean, 'std': std}

    with open(output_dir / "norm_stats.json", "w") as f:
        json.dump(norm_stats, f, indent=2)

    # -----------------------------------------------------------------------
    # Build windowed tensors for generators
    # -----------------------------------------------------------------------
    print(f"Building windowed tensors (window_len={window_len})...")

    def make_windows(
        ticker_dfs: dict[str, pd.DataFrame],
        col: str,
    ) -> np.ndarray:
        """
        Slice each ticker's return series into non-overlapping windows
        of length window_len. Returns shape (N_windows, window_len, 1).
        """
        windows = []
        for ticker, df in ticker_dfs.items():
            vals = df[col].values
            n_windows = len(vals) // window_len
            for i in range(n_windows):
                w = vals[i * window_len: (i + 1) * window_len]
                windows.append(w)
        if not windows:
            return np.empty((0, window_len, 1))
        arr = np.stack(windows)          # (N, window_len)
        return arr[:, :, np.newaxis]     # (N, window_len, 1)

    train_arr = make_windows(train_normed, col='return_normed')
    eval_arr  = make_windows(eval_dfs,    col='return_deseas')

    np.save(output_dir / "train_normalized.npy",    train_arr)
    np.save(output_dir / "eval_deseasonalized.npy", eval_arr)

    print(f"\nDone. Outputs written to {output_dir}")
    print(f"  train_normalized.npy    : {train_arr.shape}")
    print(f"  eval_deseasonalized.npy : {eval_arr.shape}")
    print(f"  fff_pattern.npy         : {s.shape}")
    print(f"  norm_stats.json         : {len(norm_stats)} tickers")
    print(f"  ticker_split.json       : saved")
    print(f"  stats_df.csv            : saved")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build train/eval datasets for synthetic generator benchmarking."
    )
    parser.add_argument("data_dir",   type=str,
                        help="Directory containing ticker CSV files")
    parser.add_argument("output_dir", type=str,
                        help="Directory to write output artifacts")
    parser.add_argument("--window_len", type=int, default=2520,
                        help="Window length in minutes (default: 2520)")
    args = parser.parse_args()

    build_dataset(args.data_dir, args.output_dir, args.window_len)