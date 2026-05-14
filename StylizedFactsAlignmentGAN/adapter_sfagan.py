"""
adapter_sfagan.py
=================
Connects the canonical data pipeline (data_prep.py) to the SFAGan
training loop (train.py) without modifying either.

What this file does
-------------------
1.  Loads ``train_normalized.npy`` — shape (N_windows, 2520, 1),
    per-ticker z-scored deseasonalized returns, produced by data_prep.py.

2.  Splits windows into train (90%) and validation (10%) using a
    temporal-style split: first 90% of windows are training, last 10%
    are validation. This preserves the time-ordering within each ticker's
    contribution and mirrors the val_tail_frac=0.10 used in the original
    data.py pipeline.

3.  Wraps the training windows in ``CanonicalWindowDataset``, a minimal
    PyTorch Dataset that replaces ``RandomCropMultiTickerDataset``.
    Instead of random-cropping within long per-ticker series, it randomly
    samples one of the 8666 pre-computed canonical windows each
    __getitem__. This exposes the generator to ~8666 unique windows per
    pass, compared to ~1000 in the original 50-ticker CSV pipeline.

4.  Produces a fixed val_real tensor of shape (N_val, T, 1) for use
    by train_sfag()'s convergence check.

5.  After training, calls ``sample_synthetic()`` to draw M=200
    independent synthetic windows from the best generator checkpoint,
    inverse-normalizes them using the population mean/std of the
    training corpus, and saves the result as ``sfagan_synthetic.npy``
    of shape (200, 2520, 1). This is the file the evaluation framework
    consumes.

Why no changes to train.py or data.py
--------------------------------------
train_sfag() takes (dataloader, val_real) as its first two arguments
and is otherwise data-agnostic. The adapter produces exactly those two
objects from canonical numpy arrays, leaving all training logic intact.

Design decisions
----------------
- Val split is index-based (first 90% train, last 10% val), not
  random, so the split is deterministic and reproducible without a seed.
- The Dataset samples with replacement (each __getitem__ draws a random
  window index), so n_per_epoch controls gradient update frequency
  independently of corpus size — same as the original design.
- Inverse-normalization uses the population mean and std across all
  8666 training windows (not per-ticker stats), because the generator
  is ticker-agnostic and has no per-ticker identity to invert.
- M=200 synthetic windows matches our evaluation protocol
  (200 windows x 2520 minutes = 504,000 total synthetic returns).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CanonicalWindowDataset(Dataset):
    """
    Wraps the pre-computed training windows from train_normalized.npy.

    Each __getitem__ draws one window uniformly at random from the pool.
    This is equivalent to random-cropping across a large multi-ticker
    corpus, because each window was itself drawn from a different part of
    a different ticker's return series.

    Parameters
    ----------
    windows     : np.ndarray, shape (N, T, 1), float32, already normalized
    n_per_epoch : int, number of samples returned per epoch
    """

    def __init__(self, windows: np.ndarray, n_per_epoch: int = 4000):
        assert windows.ndim == 3 and windows.shape[2] == 1, \
            f"Expected shape (N, T, 1), got {windows.shape}"
        self.windows     = torch.from_numpy(windows.astype(np.float32))
        self.N           = len(windows)
        self.n_per_epoch = n_per_epoch

    def __len__(self) -> int:
        return self.n_per_epoch

    def __getitem__(self, idx: int) -> torch.Tensor:
        i = np.random.randint(0, self.N)
        return self.windows[i]          # (T, 1)


# ---------------------------------------------------------------------------
# Main: build dataloader + val_real
# ---------------------------------------------------------------------------

def build_sfagan_data(
    train_npy_path: str | Path,
    val_frac:       float = 0.10,
    batch_size:     int   = 64,
    n_per_epoch:    int   = 4000,
    num_workers:    int   = 0,
) -> tuple[DataLoader, torch.Tensor, dict]:
    """
    Load canonical training windows and produce (dataloader, val_real).

    Parameters
    ----------
    train_npy_path : path to train_normalized.npy
    val_frac       : fraction of windows held out for validation
    batch_size     : DataLoader batch size
    n_per_epoch    : samples per epoch (controls batches-per-epoch)
    num_workers    : DataLoader workers (0 = main process)

    Returns
    -------
    dataloader  : DataLoader yielding (B, T, 1) normalized windows
    val_real    : torch.Tensor of shape (N_val, T, 1)
    meta        : dict with corpus statistics for inverse-normalization
    """
    train_npy_path = Path(train_npy_path)
    assert train_npy_path.exists(), f"Not found: {train_npy_path}"

    windows = np.load(train_npy_path)          # (N, T, 1), float32 or float64
    windows = windows.astype(np.float32)
    N, T, _ = windows.shape

    print(f"Loaded train_normalized.npy : {windows.shape}")

    # --- temporal-style split: first 90% train, last 10% val ---
    n_val   = max(1, int(N * val_frac))
    n_train = N - n_val

    train_windows = windows[:n_train]          # (n_train, T, 1)
    val_windows   = windows[n_train:]          # (n_val,   T, 1)

    print(f"Train windows : {n_train}")
    print(f"Val windows   : {n_val}")

    # --- population statistics for inverse-normalization at inference ---
    # The generator is ticker-agnostic, so we record the corpus-level
    # mean and std of the training windows for inversion later.
    pop_mean = float(train_windows.mean())
    pop_std  = float(train_windows.std())

    meta = {
        "pop_mean":    pop_mean,
        "pop_std":     pop_std,
        "n_train":     n_train,
        "n_val":       n_val,
        "T":           T,
    }

    print(f"Population mean (train) : {pop_mean:.6f}")
    print(f"Population std  (train) : {pop_std:.6f}")

    # --- dataset and dataloader ---
    dataset    = CanonicalWindowDataset(train_windows, n_per_epoch=n_per_epoch)
    dataloader = DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = False,   # Dataset handles randomness in __getitem__
        drop_last   = True,
        num_workers = num_workers,
        pin_memory  = torch.cuda.is_available(),
    )

    # --- val tensor ---
    val_real = torch.from_numpy(val_windows)   # (n_val, T, 1)

    return dataloader, val_real, meta


# ---------------------------------------------------------------------------
# Inference: sample synthetic windows from trained generator
# ---------------------------------------------------------------------------

def sample_synthetic(
    checkpoint_dir:  str | Path,
    meta:            dict,
    latent_dim:      int   = 128,
    hidden_dim:      int   = 256,
    n_assets:        int   = 1,
    M:               int   = 200,
    output_path:     str | Path = "sfagan_synthetic.npy",
    device:          torch.device | None = None,
) -> np.ndarray:
    """
    Load the best generator checkpoint and sample M synthetic windows.

    Applies inverse-normalization using corpus population statistics
    so the output is in deseasonalized-return units, matching
    eval_deseasonalized.npy for the evaluation framework.

    Parameters
    ----------
    checkpoint_dir : directory containing best_G.pt (from train_sfag)
    meta           : dict returned by build_sfagan_data
    latent_dim     : must match the value used during training
    hidden_dim     : must match the value used during training
    n_assets       : must match the value used during training (1)
    M              : number of synthetic windows to sample
    output_path    : where to save sfagan_synthetic.npy
    device         : torch device (auto-detected if None)

    Returns
    -------
    synthetic : np.ndarray, shape (M, T, 1), deseasonalized returns
    """
    from generator import Generator   # local import — keeps adapter importable
                                      # without the full SFAGan package on path

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_dir = Path(checkpoint_dir)
    best_G_path    = checkpoint_dir / "best_G.pt"
    assert best_G_path.exists(), f"best_G.pt not found in {checkpoint_dir}"

    T = meta["T"]

    # Load generator
    G = Generator(latent_dim, T, n_assets, hidden_dim).to(device)
    G.load_state_dict(torch.load(best_G_path, map_location=device))
    G.eval()

    print(f"Loaded generator from {best_G_path}")
    print(f"Sampling {M} synthetic windows of length {T}...")

    # Sample in batches of 32 to avoid OOM on smaller GPUs
    batch_size = 32
    batches    = []

    with torch.no_grad():
        for start in range(0, M, batch_size):
            n   = min(batch_size, M - start)
            z   = torch.randn(n, latent_dim, device=device)
            out = G(z).cpu().numpy()              # (n, T, 1)
            batches.append(out)

    synthetic_normed = np.concatenate(batches, axis=0)   # (M, T, 1)

    # Inverse-normalize: x_raw = x_normed * pop_std + pop_mean
    pop_mean = meta["pop_mean"]
    pop_std  = meta["pop_std"]
    synthetic = synthetic_normed * pop_std + pop_mean    # (M, T, 1)

    output_path = Path(output_path)
    np.save(output_path, synthetic.astype(np.float32))

    print(f"Saved sfagan_synthetic.npy : {synthetic.shape}")
    print(f"  mean : {synthetic.mean():.6f}")
    print(f"  std  : {synthetic.std():.6f}")
    print(f"  min  : {synthetic.min():.4f}")
    print(f"  max  : {synthetic.max():.4f}")

    return synthetic


# ---------------------------------------------------------------------------
# CLI: build data only (for inspection before training)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build SFAGan dataloader from canonical data_prep outputs."
    )
    parser.add_argument("train_npy",   type=str,
                        help="Path to train_normalized.npy")
    parser.add_argument("--val_frac",  type=float, default=0.10)
    parser.add_argument("--batch_size",type=int,   default=64)
    args = parser.parse_args()

    dataloader, val_real, meta = build_sfagan_data(
        train_npy_path = args.train_npy,
        val_frac       = args.val_frac,
        batch_size     = args.batch_size,
    )

    print(f"\nDataloader batches per epoch : {len(dataloader)}")
    print(f"Val real shape               : {val_real.shape}")
    print(f"Meta                         : {meta}")

    # Smoke test: one batch
    batch = next(iter(dataloader))
    print(f"Sample batch shape           : {batch.shape}")
    print(f"Sample batch mean            : {batch.mean():.4f}")
    print(f"Sample batch std             : {batch.std():.4f}")