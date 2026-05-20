# Data pipeline

`data_prep.py` turns per-ticker intraday CSVs into numpy artifacts every generator adapter reads.

## Inputs

One CSV per ticker under `data_dir/`, with at least `timestamp` (UTC) and `close`. `download_log.csv` is ignored.

## Cleaning (steps 1–6)

1. NY regular session only: minute-of-day 570–959 (9:30–15:59).
2. Sort by time within each ticker.
3. Flag within-day gaps where the previous bar is >1 minute away.
4. Drop bars immediately after a gap (return would span multiple minutes).
5. Drop the first bar of each session day (overnight jump).
6. `log_return = log(close_t / close_{t-1})`.

## Liquidity (step 7)

| Class | Median bars / session day | Use |
|-------|---------------------------|-----|
| excluded | < 280 | dropped |
| train_only | 280–349 | training only |
| eval_eligible | ≥ 350 | eligible for eval holdout |

## Split (step 8)

20% of eval-eligible tickers → eval set (seed 42). Remaining liquid + all train_only → train.

## Deseasonalization (step 9)

Pooled FFF on training tickers: fit harmonics to mean |return| by minute-of-day τ∈[0,389], then `return_deseas = log_return / s(τ)`. Same curve applied to eval (no refit).

## Normalization (step 10)

Training tickers: per-ticker z-score of deseasonalized returns. Eval: deseasonalized only.

## Windowing

Non-overlapping windows of `window_len` minutes (default 2520 = 10 sessions). Tickers processed in sorted name order.

Eval windows get regime labels: 0 if the window’s first bar is before 2020-02-19, else 1.

## Outputs (`output_dir/`)

| File | Shape / type |
|------|----------------|
| `train_normalized.npy` | (N, T, 1) |
| `train_ticker_labels.npy` | (N,) str |
| `eval_deseasonalized.npy` | (N, T, 1) |
| `eval_ticker_labels.npy` | (N,) str |
| `eval_regime_labels.npy` | (N,) int8 |
| `benchmark_manifest.json` | T, ref std, N=200 for EF |
| `fff_pattern.npy` | (390,) |
| `*_synthetic.npy` | generator exports via `canonical.save_benchmark_corpus` |
| `norm_stats.json` | per-ticker mean/std |
| `ticker_split.json` | train / eval lists |
| `stats_df.csv` | liquidity table |

## Run

```bash
cd SyntheticGenerators   # repo root
uv run python data/data_prep.py data/raw_intraday data/output_data
# or: uv run python -m data.data_prep data/raw_intraday data/output_data
```

## Related scripts

- `regen_eval_labels.py` — rebuild eval label arrays without full pipeline
- `transform_ail.py` — AIL parquet → same window format using saved FFF
- `scripts/export_sfagan.py` — write `sfagan_synthetic.npy` from `best_G.pt` or `--from-npy`
