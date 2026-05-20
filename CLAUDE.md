# SyntheticGenerators — context for Claude Code

## What this repo does
Implements/wraps synthetic financial time series generators (SFAG, GARCH baseline,
AIL data adapter, TimeGAN, CTGAN, PARSynthesizer) behind a common sampling interface
consumed by EvaluationFramework.

## Sampling interface contract
Each generator exposes .sample(n_paths, n_steps, seed) → np.ndarray of shape
(n_paths, n_steps) containing deseasonalized log-returns matching CanonicalSchema.

## For the current task
This repo is READ-ONLY unless EvaluationFramework's resampling refactor requires
changes to the sampling interface. If a change is needed:
1. Flag it before making it
2. Keep backward compat — add new method, don't change existing signature
3. Update all generator adapters consistently

## Conventions
- Same as EvaluationFramework: uv, CanonicalSchema, explicit seeds
- Canonical artifacts from `data/data_prep.py` — see `data/README.md`
- SFAG training has known issues (mode collapse, gradient clipping) — out of scope here