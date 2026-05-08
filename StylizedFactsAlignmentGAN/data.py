import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

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