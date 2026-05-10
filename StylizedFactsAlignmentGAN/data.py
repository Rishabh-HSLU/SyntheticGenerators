import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


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