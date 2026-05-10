"""Thin wrappers over the ``arch`` package for the two SFAG-baseline GARCH models.

Two variants:

- ``vanilla_garch``  : GARCH(1,1) with Normal innovations.
  Lower-bound baseline — captures volatility clustering only.

- ``gjr_garch_t``    : GJR-GARCH(1,1,1) with Student-t innovations.
  Strong baseline — covers all four SFAG stylized facts:
    * volatility clustering (β),
    * leverage / asymmetry  (γ — the GJR term),
    * heavy tails           (Student-t innovations),
    * multi-scale persistence (high α + β + γ/2).

The fit returns ``arch_model`` results objects which expose ``.simulate()``
for synthetic-path generation in ``simulate.py``.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from arch import arch_model
from arch.univariate.base import ARCHModelResult


SCALE = 100.0   # arch fits more reliably when returns are in percent


@dataclass
class FittedGARCH:
    name: str                   # "GARCH-N" | "GJR-GARCH-t"
    result: ARCHModelResult     # the arch result object (carries .simulate)
    scale: float = SCALE        # divide simulated paths by this to recover raw scale


def fit_vanilla_garch(returns: np.ndarray) -> FittedGARCH:
    """GARCH(1,1) with zero mean and Normal innovations."""
    am = arch_model(
        returns * SCALE,
        mean="Zero",
        vol="GARCH",
        p=1, q=1,
        dist="normal",
        rescale=False,
    )
    res = am.fit(disp="off")
    return FittedGARCH(name="GARCH-N", result=res, scale=SCALE)


def fit_gjr_garch_t(returns: np.ndarray) -> FittedGARCH:
    """GJR-GARCH(1,1,1) with zero mean and Student-t innovations."""
    am = arch_model(
        returns * SCALE,
        mean="Zero",
        vol="GARCH",
        p=1, o=1, q=1,        # o=1 → GJR asymmetric term
        dist="t",
        rescale=False,
    )
    res = am.fit(disp="off")
    return FittedGARCH(name="GJR-GARCH-t", result=res, scale=SCALE)


def fit(
    returns: np.ndarray,
    variant: Literal["vanilla", "gjr_t"] = "gjr_t",
) -> FittedGARCH:
    """Convenience dispatcher."""
    if variant == "vanilla":
        return fit_vanilla_garch(returns)
    if variant == "gjr_t":
        return fit_gjr_garch_t(returns)
    raise ValueError(f"unknown variant: {variant!r}")


def summary(fitted: FittedGARCH) -> str:
    """Pretty-print fitted parameters."""
    p = fitted.result.params
    lines = [f"=== {fitted.name} ==="]
    for k, v in p.items():
        lines.append(f"  {k:<10s} = {v:+.5f}")
    lines.append(f"  loglik     = {fitted.result.loglikelihood:+.2f}")
    lines.append(f"  AIC        = {fitted.result.aic:+.2f}")
    lines.append(f"  BIC        = {fitted.result.bic:+.2f}")
    return "\n".join(lines)
