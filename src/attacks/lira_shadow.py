"""
LiRA shadow model training and evaluation — mechanical implementation.

Conceptual explanations (three delta variants, LiRA scoring formula, threat model
discussion) live in the notebooks that call these functions, not here.
"""

import gc
import json
import logging
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm as sp_norm
from sklearn.metrics import auc as sk_auc, roc_curve
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset


# ── Shadow LSTM ───────────────────────────────────────────────────────────────

class ShadowLSTM(nn.Module):
    """LSTM+FC shadow model operating on precomputed CNN feature maps (B, 32, 128)."""

    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=2, batch_first=True)
        self.fc   = nn.Linear(64, 2)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(feats)
        return self.fc(out[:, -1, :])

    def similarity(self, feats: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return torch.softmax(self.forward(feats), dim=1)[:, 1]


# ── Delta helpers ─────────────────────────────────────────────────────────────

def logit_fn(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def all_deltas(same_scores, diff_scores):
    """
    Three delta variants from per-pair P(different person) scores (softmax[:, 1]).
    Returns (raw, logit_first, mean_first) or (None, None, None) if input is empty.
    Formulas are documented in notebook 05a.
    """
    if same_scores is None or len(same_scores) == 0 or len(diff_scores) == 0:
        return None, None, None
    raw         = float(same_scores.mean() - diff_scores.mean())
    logit_first = float(logit_fn(same_scores).mean() - logit_fn(diff_scores).mean())
    mean_first  = float(logit_fn(same_scores.mean()) - logit_fn(diff_scores.mean()))
    return raw, logit_first, mean_first


def compute_subject_scores(model, feats, y_labels, mask, batch_size=512):
    """Run shadow model on one subject's pairs → (same_scores, diff_scores) as float32 arrays."""
    fm = feats[mask].float()
    ym = y_labels[mask]
    if fm.shape[0] == 0:
        return None, None
    model.eval()
    pair_scores = []
    with torch.no_grad():
        for s in range(0, len(fm), batch_size):
            pair_scores.append(model.similarity(fm[s:s + batch_size]).numpy())
    pair_scores = np.concatenate(pair_scores)
    same = pair_scores[ym == 1]  # P(different person) for y=1 (different-labeled) pairs — HIGH
    diff = pair_scores[ym == 0]  # P(different person) for y=0 (same-labeled) pairs — LOW
    return (same, diff) if len(same) > 0 and len(diff) > 0 else (None, None)


# ── Shadow training ───────────────────────────────────────────────────────────

def train_shadow_models(
    all_feats, y_tr, pair_subjects, member_list,
    te_feats, y_te, te_pair_subjects, nonmember_list,
    K=32, epochs=3,
    init_mode='target',
    split_mode='blind',
    target_state=None,
    checkpoint_path=None,
    pair_subjects_w2=None,
    batch_size=512,
    device='cpu',
    log=None,
):
    """
    Train K shadow LSTMs and collect OUT-of-bag delta scores for every subject.

    init_mode='target'    → each shadow initialised from the real target model weights (grey-box)
    init_mode='warm_base' → each shadow initialised from an attacker-trained surrogate model
                            (black-box warm: attacker ran their own full training, not the target's)
    init_mode='random'    → each shadow starts from random Xavier init (black-box cold)

    split_mode='blind'        → training data for each shadow = random 50% of all pairs;
                                 OUT designation is random and independent of training data.
    split_mode='member_aware' → training data = pairs from 50% of known members.
                                 When pair_subjects_w2 is provided, pairs where x2 belongs to
                                 an OUT subject are also excluded (clean OUT boundary).

    pair_subjects_w2 : array of shape (N,) giving the subject ID for window 2 of each training
                       pair (subj_win2_train from d5_attribution.npz). Used in member_aware mode
                       to prevent an OUT subject's gait from leaking into shadow training as x2.

    The three threat models compared in NB05c (all use split_mode='member_aware'):
      grey-box   : init_mode='target'    — real model weights
      BB warm    : init_mode='warm_base' — surrogate weights
      BB cold    : init_mode='random'    — random init

    target_state: dict with keys 'lstm' and 'fc' (state_dicts).
                  Required for init_mode='target' or 'warm_base'.
    checkpoint_path: path to a JSON file for checkpoint/resume. Pass None to disable.

    Returns
    -------
    out_raw, out_lf, out_mf : dict {subject_id: [delta, ...]}
        OUT-of-bag delta scores per subject across all K shadow models.
    """
    if log is None:
        log = logging.getLogger(__name__)

    device = torch.device(device)
    INCL_RATE = 0.5
    ckpt_path = Path(checkpoint_path) if checkpoint_path else None

    # Load checkpoint
    if ckpt_path and ckpt_path.exists():
        with open(ckpt_path) as f:
            ckpt = json.load(f)
        k_start = ckpt.get('k_done', 0)
        out_raw = defaultdict(list, {int(k): v for k, v in ckpt['raw'].items()})
        out_lf  = defaultdict(list, {int(k): v for k, v in ckpt['lf'].items()})
        out_mf  = defaultdict(list, {int(k): v for k, v in ckpt['mf'].items()})
        if k_start >= K:
            log.info(f'Checkpoint complete ({k_start}/{K} shadow models). Skipping training.')
            return dict(out_raw), dict(out_lf), dict(out_mf)
        log.info(f'Resuming from checkpoint: {k_start}/{K} done.')
    else:
        k_start = 0
        out_raw = defaultdict(list)
        out_lf  = defaultdict(list)
        out_mf  = defaultdict(list)
        log.info('No checkpoint found — starting from scratch.')

    def _save(k_done):
        if ckpt_path is None:
            return
        with open(ckpt_path, 'w') as f:
            json.dump({
                'k_done': k_done, 'K': K, 'init_mode': init_mode, 'split_mode': split_mode,
                'raw': {str(k): v for k, v in out_raw.items()},
                'lf':  {str(k): v for k, v in out_lf.items()},
                'mf':  {str(k): v for k, v in out_mf.items()},
            }, f)

    # Always consume both RNG draws per iteration for reproducible replay regardless of split_mode
    rng = np.random.default_rng(0)
    for _ in range(k_start):
        rng.choice(member_list, int(len(member_list) * INCL_RATE), replace=False)
        rng.random(len(y_tr))

    log.info(f'Training {K - k_start} shadow LSTMs (init={init_mode}, split={split_mode}, epochs={epochs})...')
    t0 = time.time()

    y_tr_tensor = torch.from_numpy(y_tr).long()

    for k in range(k_start, K):
        # Always consume both draws so RNG state stays consistent across split modes
        subject_draw = rng.choice(member_list, int(len(member_list) * INCL_RATE), replace=False)
        pair_draw    = rng.random(len(y_tr))

        if split_mode == 'member_aware':
            included           = set(int(x) for x in subject_draw)
            out_subjects       = set(member_list) - included
            included_pair_mask = np.isin(pair_subjects, list(included)) | (pair_subjects == -1)
            # Exclude pairs where x2 (reference) belongs to an OUT subject — prevents their
            # gait from leaking into shadow training through the reference window.
            if pair_subjects_w2 is not None:
                included_pair_mask = included_pair_mask & ~np.isin(pair_subjects_w2, list(out_subjects))
        else:  # 'blind': attacker does not know membership labels
            # out_subjects is a random designation used only for scoring — NOT linked to training data
            out_subjects       = set(int(x) for x in subject_draw)
            included_pair_mask = pair_draw < INCL_RATE

        loader = DataLoader(
            TensorDataset(
                all_feats[included_pair_mask].float(),
                y_tr_tensor[included_pair_mask],
            ),
            batch_size=batch_size, shuffle=True, num_workers=0,
        )

        shadow = ShadowLSTM().to(device)
        if init_mode in ('target', 'warm_base') and target_state is not None:
            shadow.lstm.load_state_dict(target_state['lstm'])
            shadow.fc.load_state_dict(target_state['fc'])

        opt     = Adam(shadow.parameters(), lr=0.0025, weight_decay=0.0015)
        loss_fn = nn.CrossEntropyLoss()

        for _ in range(epochs):
            shadow.train()
            for feats_batch, y_batch in loader:
                feats_batch, y_batch = feats_batch.to(device), y_batch.to(device)
                opt.zero_grad()
                loss_fn(shadow(feats_batch), y_batch).backward()
                opt.step()

        # OUT members: scored on train pairs
        for sid in out_subjects:
            same_sc, diff_sc = compute_subject_scores(
                shadow, all_feats, y_tr, pair_subjects == sid, batch_size
            )
            raw, lf, mf = all_deltas(same_sc, diff_sc)
            if raw is not None:
                out_raw[sid].append(raw)
                out_lf[sid].append(lf)
                out_mf[sid].append(mf)

        # Non-members: always OUT, scored on test pairs
        for sid in nonmember_list:
            same_sc, diff_sc = compute_subject_scores(
                shadow, te_feats, y_te, te_pair_subjects == sid, batch_size
            )
            raw, lf, mf = all_deltas(same_sc, diff_sc)
            if raw is not None:
                out_raw[sid].append(raw)
                out_lf[sid].append(lf)
                out_mf[sid].append(mf)

        del shadow, loader, opt, loss_fn
        gc.collect()

        if (k + 1) % 8 == 0 or k == k_start:
            _save(k + 1)
            elapsed = time.time() - t0
            eta     = elapsed / (k - k_start + 1) * (K - k - 1)
            log.info(f'  [{k + 1:2d}/{K}]  {elapsed / 60:.1f}min  ETA={eta / 60:.1f}min')

    _save(K)
    log.info(f'Done in {(time.time() - t0) / 60:.1f} min')
    return dict(out_raw), dict(out_lf), dict(out_mf)


# ── Evaluation ────────────────────────────────────────────────────────────────

def tpr_at_fpr(fpr_arr, tpr_arr, target):
    idx = np.searchsorted(fpr_arr, target, side='right') - 1
    return float(tpr_arr[max(0, idx)])


def lira_eval(out_dict, target_dict, member_list, nonmember_list):
    """
    Fit a per-subject Gaussian on shadow OUT-deltas, compute LiRA scores, return ROC metrics.

    Scoring formula (see notebook 05b for context):
        score(s) = Φ( δ_target(s) ; μ_out(s), σ_out(s) )

    Subjects with fewer than 5 OUT estimates are excluded (insufficient for Gaussian fit).

    Returns
    -------
    fpr, tpr, auc, tpr@0.10, tpr@0.20, scores_dict, member_arr, nonmember_arr
    """
    lira_scores = {}
    for sid in member_list + nonmember_list:
        shadow_deltas = out_dict.get(sid, [])
        if len(shadow_deltas) < 5 or sid not in target_dict:
            continue
        null_mean = float(np.mean(shadow_deltas))
        null_std  = float(np.std(shadow_deltas, ddof=1)) + 1e-6
        lira_scores[sid] = float(sp_norm.cdf(target_dict[sid], null_mean, null_std))

    m_arr  = np.array([(sid, lira_scores[sid]) for sid in member_list    if sid in lira_scores])
    nm_arr = np.array([(sid, lira_scores[sid]) for sid in nonmember_list if sid in lira_scores])
    labels     = np.concatenate([np.ones(len(m_arr)), np.zeros(len(nm_arr))])
    scores_arr = np.concatenate([m_arr[:, 1], nm_arr[:, 1]])
    fpr_arr, tpr_arr, _ = roc_curve(labels, scores_arr)
    auc_r = sk_auc(fpr_arr, tpr_arr)
    return (
        fpr_arr, tpr_arr, auc_r,
        tpr_at_fpr(fpr_arr, tpr_arr, 0.10),
        tpr_at_fpr(fpr_arr, tpr_arr, 0.20),
        lira_scores, m_arr, nm_arr,
    )
