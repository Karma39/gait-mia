# Membership Inference Attack on Gait Authentication
---

## Overview

This project investigates **Membership Inference Attacks (MIA)** on a gait authentication system.
Given only black-box access to a trained authenticator that outputs P(same person) for a pair of gait windows, the attack determines whether a specific subject's data was used to train the underlying CNN encoder.

This differs from prior work (Milani, WIFS 2024) which attacks an identification model with direct per-class logit access. Here the attacker only sees binary pair similarity scores, no class labels are available: requiring a different attack strategy.

---

## Pipeline

```
Dataset #1 (98 subjects, IDs 21-118)
        │
        ▼
[NB02] Train CNN Encoder  ──────────► checkpoints/cnn_encoder.pt   (91.77% acc)
        │
        ▼  [CNN frozen from here on]
        │
Dataset #5 (66,542 same/diff pairs, same 98 subjects)
        │
        ▼
[NB03] Train LSTM Authenticator ────► checkpoints/auth_model.pt    (AUC 0.903)
        │
        ▼
[NB04] Extract MIA Signal ──────────► logs/04_mia_scores.npz       (AUC 0.844)
  Members:     D1 IDs 21-118 (84 with D5 pairs)
  Non-members: D2 IDs  1-20  (cross-session, never trained)
        │
        ▼
[NB05] MIA Attack ──────────────────► results/05_*.png
  Simple delta:   AUC 0.844  TPR@FPR=0.10: 0.226
  LiRA (K=32):    AUC 0.885  TPR@FPR=0.10: 0.655
```

---

## Subject Split

| Group | IDs | Role |
|---|---|---|
| Members | 21-118 (98 subjects) | CNN training (D1) + LSTM training (D5) |
| Non-members | 1-20 (20 subjects) | Never used in any training step |

Non-members appear in both Dataset #1 and Dataset #2 but are excluded from all training.

---

## MIA Signal

The authenticator outputs only P(same) for a pair, no per-class logit exists. The attack uses a **per-subject delta score** (motivated by SD-MI, AAAI 2023):

```
δ(subject) = mean P(same | same-person pairs) − mean P(same | diff-person pairs)
```

Members have stronger CNN representations → larger δ (mean 0.763).  
Non-members have weaker representations → smaller δ (mean 0.503).  
Gap = 0.260, simple-delta AUC = 0.844.

LiRA (Carlini et al., S&P 2022) calibrates δ against K=32 shadow LSTM models trained with/without each subject, improving TPR@FPR=0.10 from 0.226 → 0.655.

---

## Results

| Stage | Metric | Value |
|---|---|---|
| CNN encoder | Test accuracy | 91.77% (paper ref: ~91%) |
| LSTM authenticator | AUC | 0.903 |
| LSTM authenticator | Test accuracy | 83.80% |
| Simple delta MIA | AUC | 0.844 |
| Simple delta MIA | TPR @ FPR=0.10 | 0.226 |
| LiRA mean-first (K=32) | AUC | **0.885** |
| LiRA mean-first (K=32) | TPR @ FPR=0.10 | **0.655** |

---

## Structure

```
gait_mia/
├── notebooks/
│   ├── 01_explore_datasets.ipynb       dataset audit and subject split
│   ├── 02_train_cnn_encoder.ipynb      Phase 1: 98-class CNN identification
│   ├── 03_train_authenticator.ipynb    Phase 2: frozen CNN + LSTM authentication
│   ├── 04_mia_signal.ipynb             delta score extraction and signal analysis
│   └── 05_mia_attack.ipynb             simple delta + LiRA attack with ROC curves
│
├── src/
│   ├── models/
│   │   ├── gait_cnn.py                 GaitCNN (Zou et al. 2020) — 330,466 params
│   │   └── auth_model.py               frozen CNN + 2-layer LSTM — 83,074 params
│   ├── data/
│   │   ├── dataset.py                  Dataset #1/#2 loader (6-channel, 128-step windows)
│   │   └── auth_dataset.py             Dataset #5 pair loader
│   └── utils/
│       └── latex_writer.py             auto-generate LaTeX metric macros
│
├── results/                            all figures (git-tracked)
│   ├── 01_*.png                        dataset exploration
│   ├── 02_*.png                        CNN training curves and t-SNE
│   ├── 03_*.png                        authenticator ROC and score distributions
│   ├── 04_*.png                        MIA signal analysis
│   └── 05_*.png                        attack ROC curves and LiRA comparisons
│
├── documentation/
│   ├── papers/                         reference PDFs with summaries
│   │   ├── liraAttack.pdf              Carlini et al., S&P 2022
│   │   ├── SimilarityDistributionMIA_PersonReID_AAAI2023.pdf
│   │   ├── ContinualGaitIdentification.pdf
│   │   └── UserLevelMIA_MetricEmbedding_ReID.pdf
│   └── professor_notes/                original reference code (TF1, Milani lab)
│       ├── Dataset-Description-v1.0.pdf
│       └── code/                       CNN+LSTM.py, CNN-identification.py, ...
│
├── checkpoints/                        trained model weights (gitignored)
├── logs/                               run logs, scores, LiRA checkpoint (gitignored)
├── data/                               raw datasets (gitignored — 2.6 GB)
└── latex/                              LaTeX integration (leaving aside)
```

---

## Setup

```bash
pip install torch numpy scipy scikit-learn matplotlib
```

Run notebooks in order: NB01 → NB02 → NB03 → NB04 → NB05.  
Each notebook writes `latex/generated/nbXX_metrics.tex` with `\newcommand` macros for the research.

Datasets (D1, D2, D5) are not included: contact the lab for access.

---

## References

- Carlini et al. "Membership Inference Attacks From First Principles." IEEE S&P 2022.
- Milani et al. "Gait-based User Authentication and Membership Inference." WIFS 2024.
- Zou et al. "Deep Learning-based Gait Recognition Using Smartphones." IEEE Trans. 2020.
- "Similarity Distribution based Membership Inference Attack on Person Re-ID." AAAI 2023.
