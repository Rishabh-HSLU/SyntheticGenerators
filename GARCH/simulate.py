"""Generate synthetic return paths from fitted GARCH models."""

import numpy as np

from models import FittedGARCH


def simulate_paths(
    fitted: FittedGARCH,
    n_paths: int,
    horizon: int,
    seed: int = 42,
) -> np.ndarray:
    """Simulate ``n_paths`` independent return series of length ``horizon``.

    Returns shape (n_paths, horizon, 1) on the *raw* return scale so it
    plugs straight into SFAG's AlignmentModule.
    """
    rng = np.random.default_rng(seed)
    paths = np.empty((n_paths, horizon), dtype=np.float64)

    params = fitted.result.params
    for i in range(n_paths):
        sim = fitted.result.model.simulate(
            params,
            nobs=horizon,
            initial_value=None,
            random_state=rng,
        )
        paths[i] = sim["data"].values

    paths /= fitted.scale                         # back to raw return scale
    return paths[:, :, None].astype(np.float32)   # (n_paths, horizon, 1)
