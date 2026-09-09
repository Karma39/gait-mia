# Membership Inference Attack on Gait Authentication

Investigates Membership Inference Attacks (MIA) and evasion attacks on a gait authentication system built on the whuGAIT CNN (Zou et al. 2020). 
The authenticator takes two gait windows and outputs P(different person). The MIA asks whether a subject was in the training set of the authentication LSTM; the signal also reflects CNN memorisation because a frozen encoder that has seen a subject produces tighter representations, amplifying the per-subject delta gap.

---

## Running the pipeline

```bash
# Full run on a single dataset (~2-4 h depending on hardware)
python run_pipeline.py whuGAIT
python run_pipeline.py ucihar
python run_pipeline.py wisdm
python run_pipeline.py combined        # uses frozen whuGAIT CNN encoder
python run_pipeline.py whuGAIT_signal  # CNN vs auth signal disentanglement experiment

# Run all configured datasets in sequence
python run_pipeline.py

# Smoke-test mode: reduced epochs/shadows/steps (~15-30 min on CPU)
python run_pipeline.py --quick whuGAIT
python run_pipeline.py --quick combined

# Resume from a specific notebook (e.g. after fixing a crash)
python run_pipeline.py --from 05b_mia_attack.ipynb whuGAIT

# Stop after a specific notebook
python run_pipeline.py --until 03_train_authenticator.ipynb whuGAIT

# Preview what would run without executing
python run_pipeline.py --dry-run whuGAIT

# List all configured runs
python run_pipeline.py --list
```

All hyperparameters (epochs, shadow count, PGD steps, quick-mode overrides)
live in **`config.yaml`**.

### What runs automatically

After the pipeline notebooks finish, `run_pipeline.py` always executes the
three cross-dataset analysis notebooks:

```
notebooks/analysis/compare_models.ipynb    # cross-dataset accuracy and overfitting
notebooks/analysis/compare_mia.ipynb       # cross-dataset MIA signal and LiRA
notebooks/analysis/compare_evasion.ipynb   # cross-dataset evasion results
notebooks/analysis/compare_combined.ipynb  # deep dive on the combined run
```

These aggregate results from all datasets that have completed artifacts,
generate all comparison figures, and **write the LaTeX macro files
automatically** (see LaTeX section below). They require no manual trigger and
add negligible time (~30 s total). The analysis step is skipped only when
`--until` is used, since a partial run would produce stale figures.

---

## Pipeline structure

**Standard runs** (whuGAIT / ucihar / wisdm / combined):
```
NB01  Explore + split
  └─► NB02  Train CNN encoder   [skipped for combined: reuses whuGAIT checkpoint]
        └─► NB03  Train authenticator (CNN+LSTM)
              ├─► NB04  MIA signal (per-subject delta)
              │     └─► NB05a  Delta variants (raw / logit-first / mean-first)
              │         NB05b  Grey-box LiRA (target-init shadow models)
              │         NB05c  Threat model comparison (grey / warm / cold)
              └─► NB06a  Evasion setup (epsilon budget)
                    └─► NB06b  PGD impersonation attack
                          └─► NB06c  Attack validation (spectral, K-ablation)

  [after all pipeline runs]
  ──► compare_models    cross-dataset accuracy and overfitting comparison
  ──► compare_mia       MIA signal, LiRA variants, threat model comparison
  ──► compare_evasion   evasion ASR, budget ratio, train vs test
  ──► compare_combined  deep dive on the combined run
```

**Signal disentanglement run** (whuGAIT_signal — no evasion):
```
01_explore_whuGAIT_signal   build three-group split + auth_pairs from D1
  └─► NB02  Train CNN on group A (subjects 21–40)
        └─► NB03  Train authenticator on group B (subjects 41–60, CNN frozen)
              └─► NB04  MIA signal + three-way group analysis
                    └─► NB05a / NB05b / NB05c  attack variants and threat models
                          └─► signal_disentanglement   per-group auth AUC, MIA AUC, ROC
```

---

## LaTeX integration

The analysis notebooks write macro files to `latex/generated/` automatically
at the end of every pipeline run. **No manual step is needed to update the
numbers in the thesis.**

```
latex/generated/
  compare_models_{dataset}_metrics.tex   # CNN acc, auth AUC, separation, overfit
  compare_mia_{dataset}_metrics.tex      # simple-delta AUC, LiRA AUC by variant/threat
  compare_evasion_{dataset}_metrics.tex  # ASR, budget ratio, resistant combos
```

Each file uses `\providecommand` + `\renewcommand` so multiple files can be
`\input`-ed in the same document without conflicts. The results note at
`latex/notes/results_note.tex` inputs all twelve files and falls back to `--`
for any macro not yet generated:

```latex
\input{../generated/compare_models_whuGAIT_metrics}
\input{../generated/compare_mia_whuGAIT_metrics}
\input{../generated/compare_evasion_whuGAIT_metrics}
% ... same for ucihar, wisdm, combined
\input{../generated/signal_disentanglement_metrics}   % whuGAIT_signal run only
```

To compile the results note after a run:

```bash
cd latex/notes
pdflatex results_note.tex && pdflatex results_note.tex
```

The second pass resolves cross-references. Figures are loaded directly from
`results/analysis/` and are regenerated automatically by the analysis notebooks,
so the PDF is always consistent with the latest run.

---

## Directory layout

```
run_pipeline.py              entry point
config.yaml                  all hyperparameters and quick-mode overrides
REVIEW.md                    code-review findings log

notebooks/
  01_explore_*.ipynb         data exploration and subject split (one per dataset)
  02_train_cnn_encoder.ipynb CNN identification model
  03_train_authenticator.ipynb  CNN+LSTM authentication model
  04_mia_signal.ipynb        per-subject delta computation
  05a_delta_analysis.ipynb   delta variant analysis
  05b_mia_attack.ipynb       grey-box LiRA shadow attack
  05c_shadow_comparison.ipynb threat model comparison (grey/warm/cold)
  06a_attack_setup.ipynb     epsilon budget estimation
  06b_attack.ipynb           PGD evasion attack
  06c_attack_validation.ipynb spectral analysis and K-ablation
  analysis/
    compare_models.ipynb          cross-dataset accuracy and overfitting
    compare_mia.ipynb             cross-dataset MIA signal and LiRA results
    compare_evasion.ipynb         cross-dataset evasion results
    compare_combined.ipynb        deep dive on the combined run
    signal_disentanglement.ipynb  CNN vs auth memorisation signal (whuGAIT_signal run)

artifacts/{dataset}/         structured outputs consumed by analysis notebooks
  subject_split.json         member / non-member subject IDs
  auth_pairs.npz             authentication pair arrays (X1, X2, y, subj)
  04_mia_scores.npz          per-subject delta scores
  05a_target_deltas.npz      delta variants for all subjects
  05b_per_subject_report.json  LiRA scores and predictions per subject
  05c_lira_{warm,cold}_shadows_{dataset}.json  shadow OUT-delta estimates
  06b_{dataset}_attack_results.npz  evasion results (test split)
  06b_{dataset}_train_attack_results.npz  evasion results (train split)

checkpoints/{dataset}/       model weights (gitignored)
  cnn_encoder.pt
  auth_model.pt

embeddings/{dataset}/cnn/    CNN encoder embeddings saved after NB02 (gitignored)
  train_embeddings.npy
  test_embeddings.npy

executed/{dataset}/          executed notebooks with embedded outputs
executed/analysis/           executed analysis notebooks (auto-updated)

latex/
  main.tex                   thesis LaTeX source
  notes/results_note.tex     single-column results summary (lab note)
  generated/                 auto-generated macro files (do not edit manually)

logs/{dataset}/              per-notebook training logs

results/{dataset}/           per-run figures written by pipeline notebooks
results/analysis/            cross-dataset figures written by analysis notebooks
  models_accuracy.png
  models_overfitting_curves.png
  models_separation.png
  mia_nb04_deltas.png
  mia_nb05b_lira.png
  mia_nb05c_auc.png
  mia_nb05c_accuracy.png
  mia_nb05c_threat_models.png
  mia_roc_comparison.png
  evasion_summary_bars.png
  evasion_eps_distribution.png
  evasion_train_vs_test.png
  combined_per_dataset_perf.png
  combined_score_distributions.png
  combined_training_curves.png
  signal_auth_performance.png    # auth AUC/accuracy per subject group (whuGAIT_signal)
  signal_mia_auc_per_group.png   # mean attack-score rank per group and variant
  signal_roc_per_group.png       # ROC curves for B-vs-C and A-vs-C attack settings

src/
  models/                    GaitCNN, authentication LSTM
  data/                      dataset loaders (whuGAIT, UCI-HAR, WISDM)
  attacks/                   LiRA shadow training, PGD evasion
  utils/                     latex_writer, normalisation helpers

data/                        raw datasets (gitignored, several GB)
documentation/               reference papers and original TF1 implementation
```

---

## Configured runs

| Run | Dataset | CNN encoder | Notes |
|-----|---------|-------------|-------|
| `whuGAIT` | whuGAIT | within-dataset | primary benchmark |
| `ucihar` | UCI-HAR | within-dataset | 6-class activity |
| `wisdm` | WISDM | within-dataset | 51-class activity |
| `combined` | all three | frozen whuGAIT | LSTM retrained cross-dataset |
| `whuGAIT_signal` | whuGAIT | within-dataset | CNN vs auth memorisation signal disentanglement (subjects 1–60 only; no evasion) |

---

## References

- Carlini et al. "Membership Inference Attacks From First Principles." IEEE S&P 2022.
- Milani et al. "Gait-based User Authentication and Membership Inference." WIFS 2024.
- Zou et al. "Deep Learning-based Gait Recognition Using Smartphones." IEEE Trans. 2020.
