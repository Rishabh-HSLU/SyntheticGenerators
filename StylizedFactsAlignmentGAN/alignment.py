import torch
import torch.nn as nn

def gpd_tail_index(x: torch.Tensor, q: float = 0.95) -> torch.Tensor:
    """Estimate tail index ξ via mean excess function (differentiable proxy)."""
    threshold = torch.quantile(x, q)
    exceedances = x[x > threshold] - threshold
    return exceedances.mean()  # E[X - u | X > u] ∝ ξ

def acf_squared_returns(x: torch.Tensor, K: int = 20) -> torch.Tensor:
    """ACF of squared returns up to lag K. x: (B, T, N_assets) → mean over assets."""
    x2 = x.pow(2)
    x2 = x2.mean(dim=2)                                    # (B, T)
    x2 = x2 - x2.mean(dim=1, keepdim=True)                 # demean
    var = (x2 ** 2).mean(dim=1, keepdim=True) + 1e-8       # (B, 1)
    acf = []
    for k in range(1, K + 1):
        cov = (x2[:, :-k] * x2[:, k:]).mean(dim=1)         # (B,)
        acf.append(cov / var.squeeze())
    return torch.stack(acf, dim=1)                          # (B, K)

def leverage_effect(x: torch.Tensor, horizon: int = 5) -> torch.Tensor:
    """Corr(r_t, σ_{t+1}). x: (B, T, N_assets)."""
    r = x.mean(dim=2)                                       # (B, T)
    vol = x.std(dim=2)                                      # (B, T)
    r_t   = r[:, :-horizon]                                 # (B, T-h)
    sig_t = vol[:, horizon:]                                # (B, T-h)
    # Pearson correlation per batch item
    r_t   = r_t   - r_t.mean(dim=1, keepdim=True)
    sig_t = sig_t - sig_t.mean(dim=1, keepdim=True)
    corr  = (r_t * sig_t).mean(dim=1) / (
        r_t.std(dim=1) * sig_t.std(dim=1) + 1e-8
    )
    return corr.mean()                                      # scalar


def cfvc_loss(x: torch.Tensor, windows: list = [5, 10, 20]) -> torch.Tensor:
    vols = []
    for w in windows:
        rv = x.unfold(1, w, 1).std(dim=-1).mean(dim=2)  # (B, T-w+1)
        vols.append(rv)

    # Trim all to shortest length before stacking
    min_len = min(v.size(1) for v in vols)
    vols = [v[:, :min_len] for v in vols]

    V = torch.stack(vols, dim=1)  # (B, M, T')
    V = V - V.mean(dim=2, keepdim=True)
    std = V.std(dim=2, keepdim=True) + 1e-8
    V = V / std
    corr = torch.bmm(V, V.transpose(1, 2)) / V.size(2)  # (B, M, M)
    return corr.mean(dim=0)  # (M, M)                            # (M, M)


class AlignmentModule(nn.Module):
    def __init__(self, K: int = 20, windows: list = [5, 10, 20],
                 lambda1: float = 1.0, lambda2: float = 1.0,
                 lambda3: float = 1.0, lambda4: float = 1.0):
        super().__init__()
        self.K = K
        self.windows = windows
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.lambda4 = lambda4

    def forward(self, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        # ── L_GPD ──
        real_flat = real.reshape(-1)
        fake_flat = fake.reshape(-1)
        l_gpd = (
            torch.abs(gpd_tail_index(real_flat) - gpd_tail_index(fake_flat)) +
            torch.abs(gpd_tail_index(-real_flat) - gpd_tail_index(-fake_flat))
        )

        # ── L_ACF ──
        acf_real = acf_squared_returns(real, self.K)        # (B, K)
        acf_fake = acf_squared_returns(fake, self.K)
        l_acf = ((acf_real - acf_fake) ** 2).mean()

        # ── L_Lev ──
        l_lev = torch.abs(leverage_effect(real) - leverage_effect(fake))

        # ── L_CFVC ──
        corr_real = cfvc_loss(real, self.windows)           # (M, M)
        corr_fake = cfvc_loss(fake, self.windows)
        l_cfvc = torch.norm(corr_real - corr_fake, p='fro')

        return (self.lambda1 * l_gpd  +
                self.lambda2 * l_acf  +
                self.lambda3 * l_lev  +
                self.lambda4 * l_cfvc)