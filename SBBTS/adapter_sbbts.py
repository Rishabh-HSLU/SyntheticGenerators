"""
adapter_sbbts.py
================
Connects the canonical data pipeline (data_prep.py) to the SBBTS
training interface (run_augmentation.py) without modifying either.

What this file does
-------------------
1.  Loads ``train_normalized.npy`` — shape (N_windows, 2520, 1),
    per-ticker z-scored deseasonalized returns, produced by data_prep.py.

2.  Randomly samples M_train=500 windows using a fixed seed. This is
    necessary because SBBTS loads the full trajectory tensor onto the
    GPU at once (no DataLoader). 500 windows of length 2520 requires
    ~5MB of GPU memory — well within budget.

3.  Prepends a zero initial state to each window, matching the SBBTS
    convention: X shape becomes (M_train, N+1, 1) where X[:, 0, :] = 0
    and X[:, 1:, :] = sampled returns.

4.  Computes the SBBTS internal scale: scale = X.std(dim=(0,1)) / sqrt(T),
    divides X by scale before training, and passes scale to generate_dsbm
    for automatic inversion — exactly replicating run_augmentation.py.

5.  Trains ScoreNN via training_sbbts_dsbm (beta >= 50 path, matching
    their S&P500 experiments).

6.  Generates M_simu=200 synthetic paths via generate_dsbm, which
    returns inverse-scaled paths in the same units as the input windows
    (deseasonalized normalized returns).

7.  Strips the initial zero state (column 0) from generated paths,
    giving shape (200, 2520, 1), and saves as sbbts_synthetic.npy.

Design decisions
----------------
- M_train=500: large enough for distributional diversity, small enough
  for GPU memory. Their paper used 50-500 trajectories.
- N=2520: set at ScoreNN construction time — it is a hyperparameter,
  not a structural constraint.
- Fixed seed=42 for window subsampling — reproducible.
- beta=100: same as their S&P500 experiments (triggers the DSBM path).
- All other hyperparameters match run_augmentation.py defaults.
- No changes to any SBBTS source file.

Replicating run_augmentation.py exactly
----------------------------------------
Original:
    cluster = np.load(...)                     # (M, N)
    log_returns = np.zeros((M, N+1, 1))
    log_returns[:, 1:, 0] = cluster
    X = torch.tensor(log_returns).float()
    scale = X.std(dim=(0,1)) / sqrt(T)
    X /= scale
    model, y_0 = training_sbbts_dsbm(X, ...)
    X_sbb = generate_dsbm(..., scale=scale, ...)

Adapter:
    windows = train_normalized.npy[:, :, :]    # (8666, 2520, 1)
    sample 500 → (500, 2520, 1)
    prepend zero → (500, 2521, 1)              # same as log_returns
    scale = X.std(dim=(0,1)) / sqrt(T)         # identical computation
    X /= scale                                  # identical
    model, y_0 = training_sbbts_dsbm(X, ...)   # identical call
    X_sbb = generate_dsbm(..., scale=scale, ...) # identical, auto-inverts
"""

from __future__ import annotations

import sys
from math import sqrt
from pathlib import Path

import numpy as np
import torch

_SBBTS = Path(__file__).resolve().parent
_REPO = _SBBTS.parent
for p in (_SBBTS, _REPO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


# ---------------------------------------------------------------------------
# Constants — match run_augmentation.py defaults
# ---------------------------------------------------------------------------

M_TRAIN          = 500   # windows subsampled for GPU-memory-safe full-batch training
BENCHMARK_N_PATHS = 200  # canonical benchmark corpus size (matches EvaluationFramework)
RANDOM_SEED = 42
T          = 1        # diffusion time horizon (SBBTS convention)
BETA       = 100      # noise schedule — triggers training_sbbts_dsbm path
K          = 5        # number of DSBM iterations
N_PI       = 50       # discretization steps for generation
SAFE_T     = 1e-2     # numerical stability floor

# ScoreNN architecture — match run_augmentation.py
D_MODEL    = 128
HIDDEN_DIM = 64
NHEAD      = 32
N_LAYERS   = 1

# Training
N_EPOCHS   = 1000
LR         = 1e-3
PATIENCE   = 15
DELTA      = 1e-3
# Attention memory ~ batch * nhead * L^2. At L=2520 use 1-2 on 16GB GPUs, 4 on 24GB+.
BATCH_SIZE = 2


# ---------------------------------------------------------------------------
# Step 1-3: load canonical data and build SBBTS-format tensor
# ---------------------------------------------------------------------------

def build_sbbts_tensor(
    train_npy_path: str | Path,
    M_train:        int = M_TRAIN,
    seed:           int = RANDOM_SEED,
    device:         torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, dict]:
    """
    Load canonical windows, subsample, prepend zero initial state,
    and return the SBBTS-format tensor X.

    Parameters
    ----------
    train_npy_path : path to train_normalized.npy
    M_train        : number of windows to subsample
    seed           : random seed for subsampling
    device         : torch device

    Returns
    -------
    X     : torch.Tensor, shape (M_train, N+1, 1), scaled, on device
    scale : torch.Tensor, shape (1,) or scalar, the SBBTS scale factor
    idx   : np.ndarray of sampled window indices (for reproducibility)
    meta  : population mean/std of subsampled z-scored windows (for export)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_npy_path = Path(train_npy_path)
    assert train_npy_path.exists(), f"Not found: {train_npy_path}"

    windows = np.load(train_npy_path).astype(np.float32)  # (N_total, T, 1)
    N_total, seq_len, d = windows.shape
    assert d == 1, f"Expected univariate (d=1), got d={d}"

    print(f"Loaded train_normalized.npy : {windows.shape}")

    # --- subsample M_train windows ---
    rng = np.random.default_rng(seed)
    idx = rng.choice(N_total, size=min(M_train, N_total), replace=False)
    idx = np.sort(idx)
    sampled = windows[idx]                                 # (M_train, seq_len, 1)

    print(f"Subsampled {len(idx)} windows (seed={seed})")

    # --- prepend zero initial state (SBBTS convention) ---
    # X shape: (M_train, seq_len+1, 1)
    # X[:, 0, :] = 0  (starting point of the diffusion)
    # X[:, 1:, :] = sampled returns
    M = len(idx)
    N = seq_len
    log_returns        = np.zeros((M, N + 1, 1), dtype=np.float32)
    log_returns[:, 1:, :] = sampled

    X = torch.tensor(log_returns).to(torch.float32).to(device)

    # --- SBBTS internal scale (exact replication of run_augmentation.py) ---
    scale = X.std(dim=(0, 1)) / sqrt(T)
    X    /= scale

    print(f"X shape (after prepend + scale): {X.shape}")
    print(f"Scale                           : {scale.item():.6f}")
    print(f"X std after scaling             : {X.std().item():.6f}")

    meta = {"pop_mean": float(sampled.mean()), "pop_std": float(sampled.std())}
    return X, scale, idx, meta


# ---------------------------------------------------------------------------
# Step 4-5: train SBBTS
# ---------------------------------------------------------------------------

def train_sbbts(
    X:              torch.Tensor,
    scale:          torch.Tensor,
    checkpoint_dir: str | Path = "checkpoints/sbbts_run",
    device:         torch.device | None = None,
    batch_size:     int | None = None,
    n_epochs:       int | None = None,
    allow_cpu:      bool = False,
) -> tuple:
    """
    Train the SBBTS ScoreNN model.

    Replicates the beta>=50 training path from run_augmentation.py.

    Parameters
    ----------
    X              : SBBTS-format tensor, shape (M, N+1, 1), already scaled
    scale          : scale factor from build_sbbts_tensor
    checkpoint_dir : where to save best model checkpoint
    device         : torch device

    Returns
    -------
    model : trained ScoreNN
    y_0   : initial state tensor (output of training_sbbts_dsbm)
    """
    # Local imports — keeps adapter importable without SBBTS on sys.path
    # Add SBBTS to path if needed before importing
    from models.sbbts_model import ScoreNN
    from training.training_sbbts_dsbm import training_sbbts_dsbm

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not allow_cpu:
        raise RuntimeError(
            "SBBTS training needs a GPU (seq_len=2520). "
            "Use SBBTS/training_sbbts.ipynb or scripts/train_sbbts.sh on your GPU server."
        )
    batch_size = BATCH_SIZE if batch_size is None else batch_size
    n_epochs = N_EPOCHS if n_epochs is None else n_epochs

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    M, N_plus1, d = X.shape
    N = N_plus1 - 1

    print(f"\nBuilding ScoreNN: d={d}, N={N}, d_model={D_MODEL}, "
          f"hidden_dim={HIDDEN_DIM}, nhead={NHEAD}, n_layers={N_LAYERS}")

    model = ScoreNN(
        d, D_MODEL, HIDDEN_DIM, NHEAD, N_LAYERS, N, device=device
    ).to(device)

    print(f"Training SBBTS: beta={BETA}, K={K}, n_epochs={n_epochs}, "
          f"lr={LR}, batch_size={batch_size}, patience={PATIENCE}")

    model, y_0 = training_sbbts_dsbm(
        X,
        model,
        T,
        BETA,
        K,
        lr          = LR,
        n_epochs    = n_epochs,
        safe_t      = SAFE_T,
        batch_size  = batch_size,
        patience    = PATIENCE,
        delta       = DELTA,
    )

    # Save model checkpoint
    torch.save({
        'model':   model.state_dict(),
        'y_0':     y_0,
        'scale':   scale,
        'd':       d,
        'N':       N,
    }, checkpoint_dir / "best_sbbts.pt")

    print(f"Saved checkpoint to {checkpoint_dir / 'best_sbbts.pt'}")

    return model, y_0


# ---------------------------------------------------------------------------
# Step 6-7: generate synthetic paths
# ---------------------------------------------------------------------------

def sample_synthetic(
    X:              torch.Tensor,
    model,
    y_0:            torch.Tensor,
    scale:          torch.Tensor,
    meta:           dict,
    output_dir:     str | Path,
    M_simu:         int = BENCHMARK_N_PATHS,
    output_path:    str | Path | None = None,
    device:         torch.device | None = None,
) -> np.ndarray:
    """
    Generate M_simu synthetic paths and save as sbbts_synthetic.npy.

    generate_dsbm automatically inverse-scales the output using the
    provided scale factor — output is in the same units as the input
    windows (deseasonalized normalized returns).

    The initial zero column (index 0) is stripped, giving shape
    (M_simu, N, 1) = (200, 2520, 1).

    Parameters
    ----------
    X           : SBBTS-format tensor used for training (provides reference)
    model       : trained ScoreNN
    y_0         : initial state from training_sbbts_dsbm
    scale       : scale factor from build_sbbts_tensor
    M_simu      : number of synthetic paths to generate
    output_path : where to save sbbts_synthetic.npy

    Returns
    -------
    synthetic : np.ndarray, shape (M_simu, N, 1)
    """
    from diffusion_dsbm import generate_dsbm
    try:
        from data_pipeline.canonical import save_benchmark_corpus as _save_corpus
    except ImportError:
        _save_corpus = None

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    N = X.shape[1] - 1     # strip the prepended zero column

    print(f"\nGenerating {M_simu} synthetic paths of length N={N}...")

    X_sbb = generate_dsbm(
        N,
        X,
        model,
        y_0,
        N_pi    = N_PI,
        T       = T,
        beta    = BETA,
        M_simu  = M_simu,
        N_batch = 1,
        scale   = scale,
        exp     = False,
        safe_t  = SAFE_T,
    )

    # X_sbb shape from generate_dsbm: (M_simu, N+1, d) or (M_simu, N, d)
    # Strip the initial zero column if present
    if X_sbb.shape[1] == N + 1:
        X_sbb = X_sbb[:, 1:, :]      # (M_simu, N, d)

    # Move to numpy
    if isinstance(X_sbb, torch.Tensor):
        synthetic = X_sbb.cpu().numpy().astype(np.float32)
    else:
        synthetic = np.asarray(X_sbb, dtype=np.float32)

    syn = synthetic[:, :, 0] if synthetic.ndim == 3 else synthetic
    syn = syn * meta["pop_std"] + meta["pop_mean"]
    if output_path is None:
        output_path = Path(output_dir) / "sbbts_synthetic.npy"
    if _save_corpus is not None:
        arr = _save_corpus(syn, output_path, output_dir)
    else:
        # data_pipeline not installed: save raw (no volatility alignment)
        arr = syn
        np.save(output_path, arr[:, :, np.newaxis].astype(np.float32))
    print(f"Saved {output_path}  mean={arr.mean():.6f}  std={arr.std():.6f}")
    return arr[:, :, np.newaxis]


# ---------------------------------------------------------------------------
# Convenience: load from checkpoint and re-generate
# ---------------------------------------------------------------------------

def load_and_generate(
    checkpoint_dir: str | Path,
    train_npy_path: str | Path,
    output_dir:     str | Path,
    M_simu:         int = BENCHMARK_N_PATHS,
    output_path:    str | Path | None = None,
    device:         torch.device | None = None,
) -> np.ndarray:
    """
    Load a saved SBBTS checkpoint and generate synthetic paths without
    re-training. Useful if training was done in a previous session.
    """
    from models.sbbts_model import ScoreNN

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_dir = Path(checkpoint_dir)
    ckpt_path      = checkpoint_dir / "best_sbbts.pt"
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"

    ckpt  = torch.load(ckpt_path, map_location=device)
    d     = ckpt['d']
    N     = ckpt['N']
    scale = ckpt['scale'].to(device)
    y_0   = ckpt['y_0']

    model = ScoreNN(
        d, D_MODEL, HIDDEN_DIM, NHEAD, N_LAYERS, N, device=device
    ).to(device)
    model.load_state_dict(ckpt['model'])

    print(f"Loaded checkpoint from {ckpt_path}")

    # Rebuild X for generate_dsbm reference (uses the same subsample)
    X, scale_recomputed, _, meta = build_sbbts_tensor(
        train_npy_path, M_train=M_TRAIN, seed=RANDOM_SEED, device=device
    )

    return sample_synthetic(
        X, model, y_0, scale, meta, output_dir,
        M_simu=M_simu, output_path=output_path, device=device,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Train SBBTS and generate synthetic financial time series."
    )
    parser.add_argument("train_npy", type=str, help="train_normalized.npy")
    parser.add_argument(
        "output_dir", type=str, help="data/output_data (benchmark reference)"
    )
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/sbbts_run")
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--M_train",        type=int, default=M_TRAIN)
    parser.add_argument("--M_simu", type=int, default=BENCHMARK_N_PATHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--n_epochs", type=int, default=N_EPOCHS)
    parser.add_argument(
        "--load_only",
        action="store_true",
        help="Skip training, load checkpoint and generate only",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Smoke tests only; full training needs CUDA",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.load_only:
        load_and_generate(
            checkpoint_dir=args.checkpoint_dir,
            train_npy_path=args.train_npy,
            output_dir=args.output_dir,
            M_simu=args.M_simu,
            output_path=args.output_path,
            device=device,
        )
    else:
        X, scale, idx, meta = build_sbbts_tensor(
            train_npy_path=args.train_npy,
            M_train=args.M_train,
            seed=RANDOM_SEED,
            device=device,
        )

        model, y_0 = train_sbbts(
            X=X,
            scale=scale,
            checkpoint_dir=args.checkpoint_dir,
            device=device,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            allow_cpu=args.allow_cpu,
        )

        sample_synthetic(
            X, model, y_0, scale, meta, args.output_dir,
            M_simu=args.M_simu,
            output_path=args.output_path,
            device=device,
        )