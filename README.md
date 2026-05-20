# SyntheticGenerators

Wrappers and adapters for synthetic financial time series models, all fed from one canonical data pipeline (`data/data_prep.py`).

## Layout

```
data/           raw CSVs, data_prep, output artifacts
GARCH/          GJR-GARCH baseline + adapter
StylizedFactsAlignmentGAN/   SFAG training + adapter
SBBTS/          score-based diffusion + adapter
scripts/        utilities
```

## Sampling contract

Each generator adapter exposes sampling compatible with EvaluationFramework
(`~/PycharmProjects/EvaluationFramework`):

```python
.sample(n_paths, n_steps, seed) -> np.ndarray  # shape (n_paths, n_steps)
```

Deseasonalized log-returns on the canonical grid. Do not change existing signatures without a backward-compatible alias.

## Quick start

1. Build canonical data (see `data/README.md`).
2. Train or run the generator you care about via its adapter (`adapter_*.py`).
3. Adapters write `data/output_data/{garch,sfagan,sbbts,ail}_synthetic.npy` via `data/canonical.py` (200 paths, eval-matched std).
4. Run EvaluationFramework `bench.py` for ranked comparison.

## Documentation

Detailed changelog and rationale for the benchmark pipeline refactor:  
[`docs/BENCHMARK_PIPELINE_CHANGES.md`](docs/BENCHMARK_PIPELINE_CHANGES.md)

LaTeX documentation from code (LLM + optional PDF):  
[`docs/latex/README.md`](docs/latex/README.md) · `scripts/docgen.py`

## Dependencies

Python ≥3.12, managed with `uv` (`pyproject.toml`).
