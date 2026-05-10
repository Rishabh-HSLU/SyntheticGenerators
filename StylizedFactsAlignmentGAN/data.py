import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ──────────────────────────────────────────────────────────────────────────────
# Multi-ticker helpers (50-ticker training to fix data starvation)
# ──────────────────────────────────────────────────────────────────────────────

def _load_returns_from_csv(csv_path: str) -> np.ndarray:
    """Read close prices from a CSV and return the log-return series."""
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["returns"] = np.log(df["close"] / df["close"].shift(1))
    return df["returns"].dropna().values.astype(np.float32)


def select_random_tickers(data_dir: str, n: int = 50, seed: int = 42,
                          exclude: tuple = ("download_log.csv",)) -> list:
    """Pick ``n`` random ticker filenames from a directory of CSVs."""
    rng = np.random.default_rng(seed)
    files = sorted(
        f for f in os.listdir(data_dir)
        if f.endswith(".csv") and f not in exclude
    )
    n = min(n, len(files))
    return list(rng.choice(files, size=n, replace=False))


def load_multi_ticker_returns(data_dir: str, ticker_list: list,
                              val_tail_frac: float = 0.10) -> list:
    """Load and per-ticker-normalize returns for a list of tickers.

    For each ticker, the last ``val_tail_frac`` of returns is held out for
    validation. Both train and val portions are normalized with the **train
    statistics only** (no leakage).

    Returns a list of ``(train_returns, val_returns)`` tuples, both
    z-scored float32 1-D arrays.
    """
    out = []
    skipped = 0
    for fname in ticker_list:
        try:
            r = _load_returns_from_csv(os.path.join(data_dir, fname))
        except Exception:
            skipped += 1
            continue
        if len(r) < 50:
            skipped += 1
            continue
        cutoff = int(len(r) * (1.0 - val_tail_frac))
        r_train, r_val = r[:cutoff], r[cutoff:]
        mu  = float(r_train.mean())
        sig = float(r_train.std() + 1e-8)
        r_train = (r_train - mu) / sig
        r_val   = (r_val   - mu) / sig
        out.append((r_train.astype(np.float32), r_val.astype(np.float32)))

    print(f"Tickers loaded : {len(out)} / {len(ticker_list)} (skipped {skipped})")
    return out


class RandomCropMultiTickerDataset(Dataset):
    """Random-crop dataset across many tickers.

    Each ``__getitem__`` (1) picks a random ticker, (2) picks a random T-length
    crop within that ticker. Combined with 50 tickers × ~50K returns each,
    this exposes the GAN to ~2.5M+ unique start positions — eliminating
    single-ticker data starvation.
    """

    def __init__(self, train_returns_list: list, T: int = 2520,
                 n_per_epoch: int = 2000):
        self.tickers = [r for r in train_returns_list if len(r) > T]
        if not self.tickers:
            raise ValueError(f"no ticker has > T={T} samples after filtering")
        self.T           = T
        self.n_per_epoch = n_per_epoch
        print(f"Train tickers  : {len(self.tickers)} / {len(train_returns_list)} usable (T={T})")

    def __len__(self) -> int:
        return self.n_per_epoch

    def __getitem__(self, idx: int) -> torch.Tensor:
        i      = np.random.randint(len(self.tickers))
        r      = self.tickers[i]
        start  = np.random.randint(0, len(r) - self.T + 1)
        return torch.from_numpy(r[start : start + self.T]).unsqueeze(-1)


def make_multi_ticker_val_tensor(val_returns_list: list,
                                 T: int = 2520) -> torch.Tensor:
    """Carve each ticker's val portion into non-overlapping (T, 1) windows
    and concatenate. Returns shape ``(N_total, T, 1)``."""
    out = []
    for r in val_returns_list:
        n = len(r) // T
        if n == 0:
            continue
        out.append(r[: n * T].reshape(n, T, 1))
    if not out:
        raise ValueError(f"no val windows produced (need val tail >= T={T})")
    arr = np.concatenate(out, axis=0).astype(np.float32)
    return torch.from_numpy(arr)


# ──────────────────────────────────────────────────────────────────────────────
# Single-ticker helpers
# ──────────────────────────────────────────────────────────────────────────────

class RandomCropReturnDataset(Dataset):
    """On-the-fly random-crop dataset for single-ticker training.

    Each ``__getitem__`` picks a uniformly-random start index in
    ``[0, len(returns) - T]`` and returns that T-length slice. Compared to
    fixed-stride sliding windows, this exposes the model to ~``len(returns) - T``
    distinct windows (≈48 638 for ACM at T=2520) without artificially
    correlated neighbours, fixing single-ticker data starvation.

    Parameters
    ----------
    returns      : 1-D float array of log-returns
    T            : window length (matches paper / SFAG sequence length)
    n_per_epoch  : number of crops returned per epoch (controls batches/epoch)
    normalize    : z-score the full series with its own (μ, σ) before cropping
    """

    def __init__(self, returns: np.ndarray, T: int = 2520,
                 n_per_epoch: int = 2000, normalize: bool = True):
        r = np.asarray(returns, dtype=np.float32).flatten()
        if len(r) <= T:
            raise ValueError(f"need len(returns) > T, got {len(r)} <= {T}")

        if normalize:
            mu  = float(r.mean())
            sig = float(r.std() + 1e-8)
            r   = (r - mu) / sig
            self.mean_std = (mu, sig)
        else:
            self.mean_std = (0.0, 1.0)

        self.returns      = r
        self.T            = T
        self.n_per_epoch  = n_per_epoch
        self._max_start   = len(r) - T

    def __len__(self) -> int:
        return self.n_per_epoch

    def __getitem__(self, idx: int) -> torch.Tensor:
        start  = np.random.randint(0, self._max_start + 1)
        window = self.returns[start : start + self.T]
        return torch.from_numpy(window).unsqueeze(-1)   # (T, 1)


def make_val_tensor(returns: np.ndarray, T: int = 2520,
                    normalize_with: tuple = None) -> torch.Tensor:
    """Carve a 1-D return series into non-overlapping (T,1) windows.

    Returns shape ``(N, T, 1)`` for use as a fixed validation batch.
    Pass ``normalize_with=(mu, sigma)`` to apply training-set statistics
    so val and train share the same scale (no leakage).
    """
    r = np.asarray(returns, dtype=np.float32).flatten()
    if normalize_with is not None:
        mu, sig = normalize_with
        r = (r - mu) / (sig + 1e-8)

    n = len(r) // T
    if n == 0:
        raise ValueError(f"val series too short: len={len(r)}, T={T}")
    r = r[: n * T].reshape(n, T, 1)
    return torch.from_numpy(r)


class ReturnSeriesDataset(Dataset):
    def __init__(self, csv_path: str, T: int = 2520):
        df = pd.read_csv(csv_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Log returns from close prices
        df["returns"] = np.log(df["close"] / df["close"].shift(1))
        df = df.dropna(subset=["returns"])

        returns = df["returns"].values.astype(np.float32)  # (N,)

        # Non-overlapping windows → (n_samples, T, 1)
        n_samples = len(returns) // T
        returns   = returns[: n_samples * T]               # trim remainder
        self.samples = returns.reshape(n_samples, T, 1)    # (n_samples, T, 1)

        # Per-window z-score normalization
        mean = self.samples.mean(axis=1, keepdims=True)
        std  = self.samples.std(axis=1, keepdims=True) + 1e-8
        self.samples = (self.samples - mean) / std

        print(f"Total windows : {n_samples}")
        print(f"Sample shape  : {self.samples[0].shape}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx])             # (T, 1)


def get_dataloaders(
    csv_path: str,
    T: int            = 2520,
    val_split: float  = 0.1,
    batch_size: int   = 24,
):
    dataset = ReturnSeriesDataset(csv_path, T=T)

    # Temporal split
    val_size   = max(1, int(len(dataset) * val_split))
    train_size = len(dataset) - val_size

    train_ds = torch.utils.data.Subset(dataset, range(train_size))
    val_ds   = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, drop_last=False)

    # Collapse val into single tensor for convergence check in train.py
    val_real = torch.cat([x for x in val_loader], dim=0)  # (N_val, T, 1)

    print(f"Train samples : {train_size}")
    print(f"Val samples   : {val_size}")
    print(f"Val tensor    : {val_real.shape}")

    return train_loader, val_real