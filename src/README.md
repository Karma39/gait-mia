# src

Everything the notebooks import lives here. The four subdirectories map onto the pipeline's concerns: `models/` has the two PyTorch modules, `data/` has the dataset-specific loaders plus the unified interface the notebooks actually call, `attacks/` has the MIA and evasion implementations, and `utils/` handles config and LaTeX output.

---

## src/models

### gait_cnn.py

The CNN encoder is a direct port of Zou et al.'s whuGAIT architecture from TensorFlow 1 to PyTorch. Input shape is `(batch, 6, 128)`: six IMU channels (three accelerometer, three gyroscope), 128 timesteps. The four convolution blocks produce a 2048-dim flat embedding via `encode()`, but the authenticator never calls that method. It calls `get_feature_maps()` instead, which returns the last conv block's output reshaped to `(batch, 16, 128)` to preserve temporal structure for the downstream LSTM. Both methods share the private `_backbone()` method so the conv stack isn't duplicated. `forward()` runs the full identification head (linear layer over `encode()`); that's only used during CNN training in NB02.

### auth_model.py

The siamese authenticator takes two gait windows and returns logits for `[same person, different person]`. Both windows pass through the same CNN encoder, the feature maps are concatenated along the time axis to give a `(batch, 32, 128)` sequence, and a 2-layer LSTM reads the sequence and collapses it to a single 64-dim hidden state. Whether the CNN is frozen or not is the caller's responsibility; NB03 calls `param.requires_grad_(False)` on the encoder before training begins, and the class itself is agnostic to that choice. The convenience method `similarity()` returns `softmax[:, 1]` as a no-grad probability scalar, which is what the MIA and attack notebooks use for scoring rather than the raw logits.

---

## src/data

### dataset.py

The whuGAIT CNN training loader. `load_dataset()` reads the original Dataset #5 directory structure and returns `(X, y_subject)` where `y_subject` holds 1-indexed subject IDs. These are identification labels (which person), not pair labels (same/different). `normalize()` computes per-channel z-score statistics from whatever array you pass it; always compute on the training split and pass the stats explicitly to the test split to avoid contamination.

### auth_dataset.py

The whuGAIT authentication loader. Dataset #5 stores each pair as a 256-column row (two windows concatenated), which `load_auth_dataset()` splits at the midpoint into `X1` and `X2` of shape `(N, 6, 128)`. The raw pair labels are `{1, 2}` (same/different); `_load_auth_y()` converts them to `{0, 1}` to match the rest of the pipeline. `normalize_auth()` works like `normalize()` in dataset.py but stacks both windows jointly before computing channel statistics, so `X1` and `X2` share the same scale. `AuthPairDataset` and `build_auth_loaders()` wrap the arrays into PyTorch DataLoaders for NB03 training.

### pair_builder.py

`build_auth_pairs()` takes any windowed dataset and produces the balanced pair inventory that NB01 writes to `auth_pairs.npz`. Same-person pairs are sampled by picking a subject at random and drawing two of their windows without replacement; different-person pairs draw two subjects without replacement and one window from each. Subject IDs are tracked for both windows in every pair, which is what lets `pairs_loader.py` later attribute each pair to a specific subject for the MIA analysis.

### ucihar_dataset.py

UCI HAR has its own pre-defined train/test split where the test set contains entirely different subjects from the training set, which maps directly onto the member/non-member split used for the MIA. The one complication is that UCI HAR includes static activities (sitting, standing, lying) mixed in with the gait windows. These are excluded with `gait_only=True` (the default) so the window distribution matches whuGAIT. `train_val_split()` builds a within-subject validation split from the train portion because NB02 needs a validation set but can't use the test subjects, since those are the held-out non-members.

### wisdm_dataset.py

WISDM stores raw timestamped CSV data per subject rather than pre-segmented windows, so `load_wisdm()` parses the files and slides a 128-sample window with 50% overlap. Accelerometer and gyroscope files are paired by subject ID and concatenated along the channel axis to produce `(N, 6, 128)`, matching the other datasets. Only walking activity (label 'A') is kept by default. Accelerometer and gyroscope timestamps can differ by a few samples across files, so the arrays are trimmed to the shorter length before segmenting.

### pairs_loader.py

The unified interface all notebooks use instead of calling dataset-specific loaders directly. `load_auth_pairs()` routes to the right source by dataset name: whuGAIT loads from raw files via `auth_dataset.py`, everything else loads from the pre-built `auth_pairs.npz`. `load_attribution()` handles the more subtle case of per-window subject attribution. For whuGAIT, this requires the `d5_attribution.npz` fingerprint lookup built in NB01, because Dataset #5's pair format doesn't encode subject IDs; for other datasets, subject IDs are recorded at pair-build time. `build_pair_subjects_train()` and `build_pair_subjects_test()` resolve which of the two windows in a cross-subject pair to attribute: for whuGAIT, it checks which subject is in the member or non-member set; for other datasets, `subj1` is always the subject of interest by construction.

---

## src/attacks

### lira_shadow.py

The shadow model training loop and LiRA evaluation. `ShadowLSTM` mirrors the LSTM head of `AuthModel` but operates on precomputed CNN feature maps rather than raw sensor input. This matters for speed: training K shadow models per run while also re-running the CNN forward pass would be prohibitively slow, so NB05b precomputes all feature maps once and hands them to the shadow training loop.

`train_shadow_models()` supports three init modes (`target`, `warm_base`, `random`) and two split modes (`blind`, `member_aware`), covering the grey-box, warm black-box, and cold black-box threat models compared in NB05c. The checkpoint/resume logic writes to a JSON file every 8 shadows, so a long run can be interrupted and continued without re-training completed models. RNG state is always advanced for both draws regardless of which split mode is active, which ensures that a checkpointed run produces identical results to an uninterrupted one.

`lira_eval()` fits a Gaussian on each subject's OUT-of-bag shadow deltas and computes a CDF score using that Gaussian as the null distribution. Subjects with fewer than 5 OUT-of-bag estimates are excluded rather than padded, because the Gaussian fit is unreliable with that little data.

### pgd.py

PGD evasion attack implementations for two attack surfaces. Surface A (`pgd_sensor`) perturbs the raw IMU signal with an L2 norm constraint. L2 is the right choice here: L-inf concentrates the budget into a few large per-sample spikes that look like impact artifacts in IMU data; L2 bounds total signal energy and produces perturbations that are spread across the window and resemble natural gait variation. Surface B (`pgd_repr_probe`) perturbs only the probe CNN feature maps, modelling a MITM scenario where the attacker intercepts the client's feature vector before it reaches the authenticator server.

`pgd_sensor_variable_eps()` is the batched binary-search variant where each pair in the batch has a different ε midpoint. The batched wrappers (`run_pgd_sensor_batched`, `run_pgd_repr_probe_batched`) apply the attacks in mini-batches when the full pair set doesn't fit in memory.

One naming note: `batch_psame()` returns `softmax[:, 1]`, which is P(different person), not P(same person). Attack success means that value falls below 0.5 so the model accepts the impostor.

---

## src/utils

### config_loader.py

Three concerns in one file: reading `config.yaml`, returning per-dataset directory paths, and returning per-dataset file paths. `dataset_dirs()` creates all the directories if they don't exist and returns a dict keyed by role (`logs`, `checkpoints`, `artifacts`, etc.). `dataset_files()` returns the standard file paths within those directories. The separation between dirs and files matters because not all callers need both. If `artifacts_dir` is not passed to `dataset_files()`, it falls back to `logs_dir` for backward compatibility with notebooks written before the `artifacts/` directory was separated from `logs/`.

### latex_writer.py

`write_latex_metrics()` takes a dict of metric names and values, optionally prefixes each key with the dataset name, and writes them as `\newcommand` macros to a `.tex` file. The prefix is what makes command names unique when multiple dataset files are loaded into the same LaTeX document, for example `\whuGAITauthTestAUC` vs `\uciharAuthTestAUC`. Underscores in metric keys are stripped from command names because LaTeX command names can't contain them. The file is overwritten on every call, so re-running an analysis notebook always gives the thesis the latest values.
