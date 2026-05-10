# alignment.py
import torch
import torch.nn as nn
import torch.nn.functional as F


def gpd_tail_index(x: torch.Tensor, q: float = 0.95) -> torch.Tensor:
    """
    Differentiable tail index via soft mean excess function.
    Uses smooth threshold instead of boolean indexing to preserve gradients.
    x: (N,) flattened returns
    """
    threshold = torch.quantile(x.detach(), q)               # detach threshold only
    weights   = torch.sigmoid((x - threshold) * 100)        # soft mask, differentiable
    exceedances = (x - threshold) * weights
    return exceedances.sum() / (weights.sum() + 1e-8)


def acf_squared_returns(x: torch.Tensor, K: int = 20) -> torch.Tensor:
    """
    ACF of squared returns up to lag K.
    x: (B, T, N_assets) → (B, K)
    """
    x2   = x.pow(2).mean(dim=2)                             # (B, T) mean over assets
    x2   = x2 - x2.mean(dim=1, keepdim=True)               # demean
    var  = (x2.pow(2)).mean(dim=1, keepdim=True) + 1e-8    # (B, 1) variance

    acf = torch.stack([
        (x2[:, :-k] * x2[:, k:]).mean(dim=1) / var.squeeze(1)
        for k in range(1, K + 1)
    ], dim=1)                                                # (B, K)

    return acf


def leverage_effect(x: torch.Tensor, horizon: int = 5) -> torch.Tensor:
    """
    Corr(r_t, σ_{t+horizon}) — should be negative for real returns.
    x: (B, T, N_assets) → scalar

    Volatility proxy:
    - N_assets > 1 : cross-sectional std (unbiased=False to silence dof warning).
    - N_assets == 1: |r_t| (the cross-sectional std is identically zero for a
      single asset, which previously made this term contribute nothing — the
      absolute return is the standard single-series volatility surrogate).
    """
    r = x.mean(dim=2)                                       # (B, T)

    if x.size(2) > 1:
        vol = x.var(dim=2, unbiased=False).add(1e-8).sqrt() # (B, T)
    else:
        vol = r.abs().add(1e-8)                             # (B, T) — |r_t|

    r_t   = r[:,   :-horizon]
    sig_t = vol[:, horizon:]

    r_t   = r_t   - r_t.mean(dim=1,   keepdim=True)
    sig_t = sig_t - sig_t.mean(dim=1, keepdim=True)

    r_std   = r_t.std(dim=1)
    sig_std = sig_t.std(dim=1)

    valid = (r_std > 1e-8) & (sig_std > 1e-8)
    if valid.sum() == 0:
        return torch.tensor(0.0, device=x.device)

    corr = (r_t[valid] * sig_t[valid]).mean(dim=1) / (
        r_std[valid] * sig_std[valid]
    )
    return corr.mean()


def cfvc_loss(x: torch.Tensor, windows: list) -> torch.Tensor:
    """
    Frobenius norm between cross-scale volatility correlation matrices.
    x: (B, T, N_assets) → (M, M)
    """
    vols    = [x.unfold(1, w, 1).var(dim=-1).add(1e-8).sqrt().mean(dim=2)
               for w in windows]                             # list of (B, T-w+1)
    min_len = min(v.size(1) for v in vols)
    V       = torch.stack([v[:, :min_len] for v in vols], dim=1)  # (B, M, T')

    V = V - V.mean(dim=2, keepdim=True)
    V = V / (V.std(dim=2, keepdim=True) + 1e-8)
    return torch.bmm(V, V.transpose(1, 2)).mean(dim=0) / min_len  # (M, M)


class AlignmentModule(nn.Module):
    def __init__(
        self,
        K:       int   = 20,
        windows: list  = [5, 10, 20],
        lambda1: float = 1.0,   # GPD
        lambda2: float = 2.0,   # ACF  — upweighted, hardest to capture
        lambda3: float = 0.5,   # Lev  — captured early, reduce pressure
        lambda4: float = 0.2,   # CFVC — large magnitude, downweight
    ):
        super().__init__()
        self.K       = K
        self.windows = windows
        # Register as buffers so they move with .to(device)
        self.register_buffer('lambda1', torch.tensor(lambda1))
        self.register_buffer('lambda2', torch.tensor(lambda2))
        self.register_buffer('lambda3', torch.tensor(lambda3))
        self.register_buffer('lambda4', torch.tensor(lambda4))

    def forward(self, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        # GPD — both tails
        l_gpd = (
            torch.abs(gpd_tail_index(real.reshape(-1))  - gpd_tail_index(fake.reshape(-1))) +
            torch.abs(gpd_tail_index(-real.reshape(-1)) - gpd_tail_index(-fake.reshape(-1)))
        )

        # ACF of squared returns
        l_acf = F.mse_loss(
            acf_squared_returns(fake, self.K),
            acf_squared_returns(real, self.K),
        )

        # Leverage effect
        l_lev = torch.abs(leverage_effect(real) - leverage_effect(fake))

        # Cross-scale volatility correlation
        l_cfvc = torch.norm(
            cfvc_loss(real, self.windows) - cfvc_loss(fake, self.windows),
            p='fro'
        )

        return (self.lambda1 * l_gpd  +
                self.lambda2 * l_acf  +
                self.lambda3 * l_lev  +
                self.lambda4 * l_cfvc)