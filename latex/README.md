# LaTeX integration

Every number in the thesis comes from the pipeline. The analysis notebooks write `\newcommand` macros to `latex/generated/`, and `main.tex` pulls them in with `\input`. Change the training config, re-run the analysis notebooks, and every table and sentence in the thesis updates on the next `latexmk` run.

---

## Workflow

1. Run the full pipeline for all datasets.
2. Open and run each analysis notebook in `notebooks/analysis/`.
3. Each notebook's final cell writes `.tex` files to `latex/generated/`.
4. Add `\input` lines in `main.tex` (or a dedicated `macros.tex`) to load them.

The generated files are committed to git so the thesis compiles on any machine without having to re-run the notebooks first.

---

## Generated files

| File | Written by |
|------|-----------|
| `latex/generated/compare_models_{ds}_metrics.tex` | `compare_models.ipynb` |
| `latex/generated/compare_mia_{ds}_metrics.tex` | `compare_mia.ipynb` |
| `latex/generated/compare_evasion_{ds}_metrics.tex` | `compare_evasion.ipynb` |

`{ds}` is one of `whuGAIT`, `ucihar`, `wisdm`, `combined`.

---

## Macro naming

Each macro is `\{prefix}{Key}` where `prefix` is the dataset name and `Key` is the metric name (camelCase).

Example:

```latex
\whuGAITauthTestAUC        % authenticator test AUC for whuGAIT
\uciharcomboASR            % evasion combo ASR for ucihar
\whuGAITsimplDAUC          % Simple-D MIA AUC for whuGAIT
```

All values are pre-formatted strings (e.g. `0.874`, `95.06\%`, `1.000`). Missing values render as `—`.

---

## Available macros per file

### `compare_models_{ds}_metrics.tex`

| Macro key | Description |
|-----------|-------------|
| `cnnTestAcc` | CNN test accuracy (%) |
| `authTestAcc` | Authenticator test accuracy (%) |
| `authTrainAcc` | Authenticator train accuracy (%) |
| `authTestAUC` | Authenticator test AUC |
| `authOverfitAcc` | Train − test accuracy gap |
| `authSameMean` | Mean P(diff) for same-person pairs |
| `authDiffMean` | Mean P(diff) for different-person pairs |
| `authSep` | Score separation (diff mean − same mean) |

### `compare_mia_{ds}_metrics.tex`

| Macro key | Description |
|-----------|-------------|
| `simplDAUC` | Simple-D MIA AUC |
| `simplDDeltaGap` | Member vs non-member delta gap |
| `{variant}ACC` | LiRA accuracy for variant (e.g. `greyLiraRawACC`) |
| `{variant}AUC` | LiRA AUC for variant |

Variants: `greyLiraRaw`, `greyLiraLf`, `greyLiraMf`, `warmLiraRaw`, ..., `coldLiraMf`.

### `compare_evasion_{ds}_metrics.tex`

| Macro key | Description |
|-----------|-------------|
| `epsTarget` | Attack budget ε_target |
| `comboASR` | Combo attack success rate |
| `budgetRatio` | Mean ε_min / ε_target (resistant pairs excluded) |
| `nResistant` | Number of resistant pairs |
| `nPairs` | Total pairs attacked |
| `nFullCombos` | Combos where every pair succeeded |
| `nCombos` | Total combos |
| `trainComboASR` | Training-split combo ASR |
| `trainBudgetRatio` | Training-split budget ratio |

---

## Including in `main.tex`

```latex
% In preamble or a dedicated macros.tex:
\input{latex/generated/compare_models_whuGAIT_metrics.tex}
\input{latex/generated/compare_models_ucihar_metrics.tex}
\input{latex/generated/compare_models_wisdm_metrics.tex}
\input{latex/generated/compare_mia_whuGAIT_metrics.tex}
\input{latex/generated/compare_evasion_whuGAIT_metrics.tex}
% ... etc.

% In text:
The authenticator achieves an AUC of \whuGAITauthTestAUC{} on whuGAIT.
The best MIA AUC is \whuGAITcoldLiraMfAUC{} (cold LiRA, magnitude-filtered).
```

---

## Regenerating macros

Re-run the three analysis notebooks after any pipeline change. Their final cells overwrite the `.tex` files in place, and `latexmk` picks up the changes on its next run.
