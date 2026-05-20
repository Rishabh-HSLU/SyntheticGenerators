# SBBTS benchmark export

Train on a **GPU** only. Sequence length 2520 makes full attention very heavy.

## Run (GPU server)

Open `training_sbbts.ipynb` from this directory (or set `SYNTHGEN_ROOT` to the repo root).

1. Restart kernel after git pull.
2. Run cell 0 (paths + CUDA check).
3. Optional: cell 2 smoke test (`batch_size=2`, 2 epochs).
4. Run cell 1 full train + export → `data/output_data/sbbts_synthetic.npy`.

If CUDA OOM: set `BATCH_SIZE = 1` in cell 1.

Do **not** use `adapter_sbbts.py` on CPU.

## Memory rule of thumb

Attention ~ `batch × 32 heads × 2520²`. `batch_size=128` needs ~100GB+ and will OOM everywhere.
