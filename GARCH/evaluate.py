"""Evaluate GARCH-simulated paths with the SFAG stylized-fact metrics.

Reuses ``StylizedFactsAlignmentGAN.alignment.AlignmentModule`` so GARCH and
SFAG numbers are directly comparable.
"""

import os
import sys
from typing import Dict

import numpy as np
import torch

# Allow importing SFAG's alignment module without changing sys.path globally
_SFAG_DIR = os.path.join(os.path.dirname(__file__), "..", "StylizedFactsAlignmentGAN")
if os.path.abspath(_SFAG_DIR) not in sys.path:
    sys.path.append(os.path.abspath(_SFAG_DIR))

from alignment import (             # noqa: E402  (sys.path mutation above)
    AlignmentModule,
    gpd_tail_index,
    acf_squared_returns,
    leverage_effect,
    cfvc_loss,
)


def _to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float32, device=device)


def evaluate(
    real_windows: np.ndarray,    # (N_real, T, 1)
    sim_windows:  np.ndarray,    # (N_sim,  T, 1)
    lambda1: float = 1.0,
    lambda2: float = 2.0,
    lambda3: float = 0.5,
    lambda4: float = 0.2,
    device:  torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    """Compute the SFAG gap and per-fact breakdown for simulated vs real windows."""
    real = _to_tensor(real_windows, device)
    sim  = _to_tensor(sim_windows,  device)

    align = AlignmentModule(
        lambda1=lambda1, lambda2=lambda2,
        lambda3=lambda3, lambda4=lambda4,
    ).to(device)

    with torch.no_grad():
        # Per-stylized-fact diagnostics
        gpd_real = gpd_tail_index(real.reshape(-1)).item()
        gpd_sim  = gpd_tail_index(sim.reshape(-1)).item()

        acf_real = acf_squared_returns(real).mean(dim=0).cpu().numpy()
        acf_sim  = acf_squared_returns(sim).mean(dim=0).cpu().numpy()

        lev_real = leverage_effect(real).item()
        lev_sim  = leverage_effect(sim).item()

        cfvc_real = cfvc_loss(real, [5, 10, 20])
        cfvc_sim  = cfvc_loss(sim,  [5, 10, 20])
        cfvc_gap  = torch.norm(cfvc_real - cfvc_sim, p="fro").item()

        sfag_gap = align(real, sim).item()

    return {
        "sfag_gap":     sfag_gap,
        "gpd_real":     gpd_real,
        "gpd_sim":      gpd_sim,
        "lev_real":     lev_real,
        "lev_sim":      lev_sim,
        "cfvc_gap":     cfvc_gap,
        "acf_mse":      float(np.mean((acf_real - acf_sim) ** 2)),
        "acf_real":     acf_real,
        "acf_sim":      acf_sim,
    }


def print_report(metrics: Dict[str, float], title: str = "GARCH Evaluation") -> None:
    """Pretty-print the evaluation dict in the same style as the SFAG notebook."""
    print("=" * 50)
    print(f"  {title}")
    print("=" * 50)
    print(f"  GPD tail index   Real: {metrics['gpd_real']:+.4f}  Sim: {metrics['gpd_sim']:+.4f}")
    print(f"  Leverage effect  Real: {metrics['lev_real']:+.4f}  Sim: {metrics['lev_sim']:+.4f}")
    print(f"  CFVC gap (Frob)      : {metrics['cfvc_gap']:.4f}")
    print(f"  ACF² MSE             : {metrics['acf_mse']:.6f}")
    print("-" * 50)
    print(f"  Total SFAG gap       : {metrics['sfag_gap']:.4f}  (lower is better)")
    print("=" * 50)
