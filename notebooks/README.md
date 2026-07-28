# Notebooks

`run_pipeline.py` handles parameter injection and saves a fully executed copy to `executed/{run_name}/` after each run. The source notebooks are never touched at runtime, so the clean version stays intact for the next run.

To step through a notebook interactively instead, open it directly from this directory. It reads `config.yaml` for defaults, but you can edit the injected cells manually before running.

---

## Running the pipeline

```bash
# Standard run
python run_pipeline.py whuGAIT

# Available run names
python run_pipeline.py --list

# Partial run
python run_pipeline.py --from 05b_mia_attack.ipynb --until 05c_shadow_comparison.ipynb whuGAIT

# Smoke-test (reduced epochs/shadows/steps, not thesis-quality)
python run_pipeline.py --quick whuGAIT
```

---

## Parameters (`config.yaml`)

All lines marked `# <<< RUNNER INJECTS THIS` in a notebook cell are overwritten by the runner at execution time.

| Parameter | config.yaml key | Quick value | Injected into |
|-----------|----------------|-------------|---------------|
| `CNN_EPOCHS` | `cnn.epochs` (25) | 2 | NB02 |
| `AUTH_EPOCHS` | `auth.epochs` (10) | 2 | NB03 |
| `AUTH_LR` | `auth.lr` (0.0025) | — | NB03 |
| `AUTH_DROPOUT` | `auth.dropout` (0.3) | — | NB03 |
| `K_LIRA` | `lira.k_shadows` (32) | 4 | NB05b, NB05c |
| `LIRA_EPOCHS` | `lira.epochs` (3) | 1 | NB05b, NB05c |
| `SURROGATE_EPOCHS` | `lira.surrogate_epochs` (5) | 1 | NB05c |
| `N_BISECT` | `pgd.n_bisect` (6) | 2 | NB06a |
| `K_PGD` | `pgd.k_pgd` (20) | 5 | NB06b |
| `N_COMBOS_TR_SAMPLE` | `pgd.n_combos_train_sample` (500) | 20 | NB06b |

`DATASET` and `ENCODER_DATASET` are always injected from the run configuration.

---

## Pipeline notebooks

### NB01: Explore + subject split

| Notebook | Dataset |
|----------|---------|
| `01_explore_datasets.ipynb` | whuGAIT |
| `01_explore_ucihar.ipynb` | ucihar |
| `01_explore_wisdm.ipynb` | wisdm |
| `01_explore_combined.ipynb` | combined |

**Writes:** `artifacts/{ds}/subject_split.json`, `artifacts/{ds}/auth_pairs.npz`

The train/held-out split has to happen before anything else, because every downstream notebook assumes a fixed membership ground truth. The main output is the authentication pair inventory: balanced same-person and different-person window pairs, with subject attribution tracked for both windows. For `combined`, training subjects come from whuGAIT; test subjects are held-out ucihar and wisdm subjects that the encoder has never seen.

---

### NB02: Train CNN encoder (`02_train_cnn_encoder.ipynb`)

**Reads:** `artifacts/{ds}/subject_split.json`  
**Writes:** `checkpoints/{ds}/cnn_encoder.pt`, `checkpoints/{ds}/cnn_encoder_meta.json`  
**Injected:** `DATASET`, `ENCODER_DATASET`, `CNN_EPOCHS`

Trains the whuGAIT CNN as a 98-class subject identifier. The useful part is the intermediate representations, not the classification head; training as an identification task turns out to produce embeddings that discriminate subjects well enough for the downstream authenticator. For cross-dataset runs where `encoder_dataset` is set, the checkpoint from the source dataset is loaded and no training occurs here.

---

### NB03: Train authenticator (`03_train_authenticator.ipynb`)

**Reads:** `artifacts/{ds}/subject_split.json`, `artifacts/{ds}/auth_pairs.npz`, `checkpoints/{ds}/cnn_encoder.pt`  
**Writes:** `checkpoints/{ds}/auth_model.pt`, `artifacts/{ds}/auth_norm_stats.npz`  
**Injected:** `DATASET`, `AUTH_EPOCHS`, `AUTH_LR`, `AUTH_DROPOUT`

The encoder stays frozen here deliberately: we're investigating what the CNN's representations leak, so letting the authenticator fine-tune the encoder during training would muddy what the MIA is actually attacking. The normalisation stats computed from the train split are saved here and reused by NB04 through NB06, ensuring a consistent scale across the whole pipeline.

---

### NB04: MIA signal (`04_mia_signal.ipynb`)

**Reads:** `artifacts/{ds}/auth_pairs.npz`, `artifacts/{ds}/auth_norm_stats.npz`, `checkpoints/{ds}/auth_model.pt`  
**Writes:** `artifacts/{ds}/04_mia_scores.npz`

This is where we check whether a naive MIA is viable before investing in LiRA. If the model memorised training subjects, their same-person scores should cluster differently from held-out subjects. The saved scores are the input to everything in NB05. Before moving on, it's worth looking at the raw distributions: if the member/non-member gap inverts or disappears here, the LiRA formulation in NB05b needs to be adjusted.

---

### NB05a: Delta analysis (`05a_delta_analysis.ipynb`)

**Reads:** `artifacts/{ds}/04_mia_scores.npz`, `artifacts/{ds}/subject_split.json`  
**Writes:** `artifacts/{ds}/05a_target_deltas.npz`

Pins down the signal direction before any shadow training. The LiRA scoring formula in NB05b assumes members score higher than non-members on the target delta, so if that assumption is wrong here the Gaussian fit would classify backwards. If the direction is inverted, you'd need to flip the CDF side in `lira_eval()` rather than fix the data.

---

### NB05b: Grey-box LiRA (`05b_mia_attack.ipynb`)

**Reads:** `artifacts/{ds}/04_mia_scores.npz`, `artifacts/{ds}/05a_target_deltas.npz`, `checkpoints/{ds}/auth_model.pt`  
**Writes:** `artifacts/{ds}/05b_lira_target_shadows_{ds}.json`, `artifacts/{ds}/05b_per_subject_report.json`  
**Injected:** `K_LIRA`, `LIRA_EPOCHS`

The main MIA experiment. Shadow models start from the target's own weights (grey-box) because the question is how much the LSTM head retains about training data, not how well an outside attacker can replicate the training process. Each shadow model is trained without a target subject's pairs, so the distribution of that subject's OUT-of-bag deltas across all K shadows characterises what the model would have predicted if they weren't a member.

---

### NB05c: Shadow comparison (`05c_shadow_comparison.ipynb`)

**Reads:** `artifacts/{ds}/04_mia_scores.npz`, `artifacts/{ds}/05a_target_deltas.npz`, `checkpoints/{ds}/auth_model.pt`  
**Writes:** `artifacts/{ds}/05c_lira_warm_shadows_{ds}.json`, `artifacts/{ds}/05c_lira_cold_shadows_{ds}.json`  
**Injected:** `K_LIRA`, `LIRA_EPOCHS`, `SURROGATE_EPOCHS`

Answers whether the grey-box advantage is real or just an artifact of starting from good weights. Cold (random init) is the standard LiRA threat model: the attacker knows the architecture but not the weights. Warm (init from an attacker-trained surrogate) is the realistic black-box scenario. Grey (target weights) is the upper bound for an informed attacker. If cold LiRA matches grey-box, the CNN's representations are leaking as much membership signal as the LSTM activations do directly.

Initialisation strategies compared:
- **Grey** (NB05b): init from target weights
- **Warm**: init from an attacker-trained surrogate model
- **Cold**: random init (standard black-box LiRA)

---

### NB06a: Attack setup (`06a_attack_setup.ipynb`)

**Reads:** `artifacts/{ds}/auth_pairs.npz`, `artifacts/{ds}/auth_norm_stats.npz`, `checkpoints/{ds}/auth_model.pt`  
**Writes:** `artifacts/{ds}/06a_{ds}_attack_setup.npz`  
**Injected:** `N_BISECT`

ε_target is set to the median L2 distance between same-subject window pairs, so each attack gets a budget calibrated to natural within-subject gait variation. An attacker who needs 10× more perturbation than any natural variation isn't mounting a practical attack. This notebook also enumerates the impostor combos: for each non-member subject, which member subjects they'll try to impersonate, and which specific pairs that involves.

---

### NB06b: PGD evasion attack (`06b_attack.ipynb`)

**Reads:** `artifacts/{ds}/06a_{ds}_attack_setup.npz`, `checkpoints/{ds}/auth_model.pt`, `artifacts/{ds}/auth_norm_stats.npz`  
**Writes:** `artifacts/{ds}/06b_{ds}_attack_results.npz`, `artifacts/{ds}/06b_{ds}_grid_ckpt.npz`, `artifacts/{ds}/06b_{ds}_train_attack_results.npz`, `artifacts/{ds}/06b_{ds}_train_grid_ckpt.npz`  
**Injected:** `K_PGD`, `N_COMBOS_TR_SAMPLE`

Finds the minimum ε that makes the authenticator accept an impostor by combining a coarse grid sweep with binary-search refinement. The two-phase approach is necessary because PGD success isn't monotone in ε: a direct binary search can land in a local-failure region and report a pair as resistant when it isn't. The training-split attack is a sanity check: if training pairs are substantially harder to fool than test pairs, that's interesting and worth reporting separately.

---

### NB06c: Attack validation (`06c_attack_validation.ipynb`)

**Reads:** `artifacts/{ds}/06b_{ds}_attack_results.npz`, `artifacts/{ds}/06a_{ds}_attack_setup.npz`, `checkpoints/{ds}/auth_model.pt`  
**Writes:** `results/{ds}/06c_*.png`

Checks whether the attack is doing something physically reasonable or just exploiting a numerical artifact. The FFT analysis shows whether perturbations concentrate at physiologically plausible frequencies. The K-ablation tells you whether more PGD steps keep helping, which matters for reporting a "converged" attack budget. The feature-space shift check is the critical one: if PGD barely moves the CNN embedding but still fools the LSTM, the vulnerability lives in the LSTM's generalisation, not the CNN encoder.

---

### NB07: Combined cross-eval (`07_combined_cross_eval.ipynb`)

**Reads:** checkpoints and artifacts from multiple datasets  
**Writes:** `results/analysis/`

Deep-dive for the combined run: breaks down authenticator performance by source dataset (ucihar vs wisdm test subjects) rather than reporting combined numbers. Not part of the standard pipeline runs; run it manually after the combined pipeline if you need per-dataset breakdown.

---

## Analysis notebooks (`analysis/`)

Run these after the pipeline finishes for all datasets. Each one loads results from `artifacts/`, computes cross-dataset comparisons, and writes LaTeX macros. Open from the `analysis/` directory or run directly in Jupyter.

### `compare_models.ipynb`
Pulls CNN accuracy and authenticator AUC/accuracy for all datasets into one view.  
**Writes:** `latex/generated/compare_models_{ds}_metrics.tex`

### `compare_mia.ipynb`
Puts Simple-D and all nine LiRA variants (grey/warm/cold × raw/Lf/Mf) side by side across datasets.  
**Writes:** `latex/generated/compare_mia_{ds}_metrics.tex`

### `compare_evasion.ipynb`
Summarises the PGD attack: budget ratio, combo ASR, and number of resistant pairs per dataset.  
**Writes:** `latex/generated/compare_evasion_{ds}_metrics.tex`

### `compare_combined.ipynb`
Breaks down the combined run by source dataset (ucihar vs wisdm test subjects separately), shows epoch curves, and checks MIA and evasion results where available.

---

## Adding a new dataset

1. Create `01_explore_{name}.ipynb` that writes `artifacts/{name}/subject_split.json` and `artifacts/{name}/auth_pairs.npz`.
2. Add a run entry to `RUNS` in `run_pipeline.py`:
   ```python
   'mydata': {
       'dataset': 'mydata',
       'encoder_dataset': None,
       'notebooks': ['01_explore_mydata.ipynb'] + PIPELINE_CORE + EVASION,
   },
   ```
3. Run `python run_pipeline.py mydata`.
