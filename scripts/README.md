# Scripts

Standalone Python scripts for tasks that run outside Jupyter: cluster jobs, macro regeneration, and multi-seed sweeps. All scripts expect to be run from the `gait_mia/` root directory.

---

## Macro regeneration

### `regenerate_compare_mia_macros.py`
Recomputes all LiRA warm/cold/grey AUC and TPR macros from stored shadow artifacts and writes them to `latex/generated/compare_mia_{ds}_metrics.tex`. Use this when NB05c has been re-run and the macro files are stale. Reproduces the exact `lira_eval` call from NB05c.

```bash
python scripts/regenerate_compare_mia_macros.py
```

### `write_06d_asr_macros.py`
Reads NB06d `.npz` artifacts and writes per-N ASR LaTeX macros for all datasets. Cross-pool (MN, NM) macros are set to `\pholder` until the corresponding `_cross.npz` files exist.

```bash
python scripts/write_06d_asr_macros.py
```

---

## Cluster-ready training scripts

These mirror their notebook counterparts but run outside Jupyter, support GPU, and resume from checkpoints. Submitted via the `.sbatch` files in `cluster/`.

### `run_lira_shadow.py`
Standalone grey-box LiRA shadow training. Replicates the NB05b pipeline for any dataset. Supports `--K`, `--epochs`, `--dropout`, `--dataset` flags.

```bash
python scripts/run_lira_shadow.py --dataset whuGAIT --K 32 --epochs 3 --dropout 0.3
```

### `run_operator_sweep.py`
Trains 6 operator LSTMs (different random seeds) and writes `artifacts/whuGAIT/08c_multiseed_results.json`. Used by NB08c to report the multi-seed MIA AUC distribution.

### `run_operator_sweep_hybrid.py`
Same as `run_operator_sweep.py` but for the hybrid key condition (2048-dim CNN code, 128-dim LSTM key).

### `run_nb06b.py`
Runs the NB06b PGD evasion attack for a single dataset outside Jupyter with GPU support and checkpoint resumption.

### `run_nb06d.py`
Runs the NB06d K-search MIA experiment for a single dataset outside Jupyter.

---

## Continual learning training scripts

### `train_cl_key2048.py`
Trains the GaitCNN in a class-incremental CL setting with a 2048-dim per-channel CDML key. Mirrors NB08a but targets `checkpoints/whuGAIT/cl8_key2048/`.

### `train_auth_key2048.py`
Trains the pairwise LSTM authenticator on top of the 2048-dim CL CNN. Mirrors NB08b.

### `train_auth_key2048_128auth.py`
Hybrid variant: 2048-dim CNN code during CL training, 128-dim broadcast key during LSTM training.

### `train_auth_key2048_phase1.py`
Phase-1 only training for the 2048-dim key (CNN CL pass without the downstream authenticator).

---

## Utilities

### `compute_eer.py`
Computes Equal Error Rate from a saved score file. Utility script used for cross-checking authentication performance against EER rather than AUC.
