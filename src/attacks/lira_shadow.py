"""
LiRA shadow model training and evaluation. Mechanical implementation only:

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

    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=2,
                            batch_first=True, dropout=dropout)
        self.drop = nn.Dropout(p=dropout)
        self.fc   = nn.Linear(64, 2)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(feats)
        return self.fc(self.drop(out[:, -1, :]))

    def similarity(self, feats: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return torch.softmax(self.forward(feats), dim=1)[:, 1]

    def get_hidden_state(self, feats: torch.Tensor) -> torch.Tensor:
        """Return last LSTM hidden state (B, 64). Dropout inactive — call after model.eval()."""
        with torch.no_grad():
            out, _ = self.lstm(feats)
            return out[:, -1, :]  # (B, 64)


# ── Delta helpers ─────────────────────────────────────────────────────────────

def logit_fn(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def all_deltas(impostor_scores, genuine_scores):
    """
    Three delta variants from per-pair P(different person) scores (softmax[:, 1]).

    impostor_scores: P(different) for y=1 (different-person) pairs — HIGH for a good model
    genuine_scores:  P(different) for y=0 (same-person) pairs    — LOW for a good model

    delta = impostor_scores.mean() - genuine_scores.mean() (positive = discriminative)

    Returns (raw, logit_first, mean_first) or (None, None, None) if input is empty.
    Formulas are documented in notebook 05a.
    """
    if impostor_scores is None or len(impostor_scores) == 0 or len(genuine_scores) == 0:
        return None, None, None
    raw         = float(impostor_scores.mean() - genuine_scores.mean())
    logit_first = float(logit_fn(impostor_scores).mean() - logit_fn(genuine_scores).mean())
    mean_first  = float(logit_fn(impostor_scores.mean()) - logit_fn(genuine_scores.mean()))
    return raw, logit_first, mean_first


def compute_subject_scores(model, feats, y_labels, mask, batch_size=512):
    """
    Run shadow model on one subject's pairs.

    Returns (impostor_scores, genuine_scores) as float32 arrays, where:
      impostor_scores: P(different person) for y=1 pairs (different-person / impostor pairs)
      genuine_scores:  P(different person) for y=0 pairs (same-person / genuine pairs)

    Label convention: y=0 = same person, y=1 = different person.
    softmax[:, 1] = P(different person); high for impostors, low for genuine pairs.
    """
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
    impostor_sc = pair_scores[ym == 1]  # y=1 = different-person pairs: P(different) is HIGH
    genuine_sc  = pair_scores[ym == 0]  # y=0 = same-person pairs:      P(different) is LOW
    return (impostor_sc, genuine_sc) if len(impostor_sc) > 0 and len(genuine_sc) > 0 else (None, None)


def compute_subject_scores_sym(
    model, feats, y_labels, subj1_arr, subj2_arr, target_sid, batch_size=512
):
    """
    Symmetric variant: collect all pairs where target_sid appears as sw1 OR sw2.

    The model is asymmetric (sw1/sw2 are different channels), so the attacker
    queries both orderings and aggregates to remove the role-specific bias.

    Returns (impostor_scores, genuine_scores) combining both directions, or (None, None).
    """
    mask = (subj1_arr == target_sid) | (subj2_arr == target_sid)
    return compute_subject_scores(model, feats, y_labels, mask, batch_size)


def compute_hidden_delta(model, feats, y_labels, mask, batch_size=512):
    """
    Centroid-distance delta in LSTM hidden space.

    For each subject, computes the mean hidden-state vector over impostor pairs (μ_imp)
    and over genuine pairs (μ_gen), then returns ‖μ_imp − μ_gen‖₂.

    A large value means the model pushes the two pair types far apart in the 64-dim
    hidden space — expected to be larger for members than non-members.

    The model must have a get_hidden_state(feats) method (ShadowLSTM or AuthModel
    via hidden_from_feats). Call after model.eval() — dropout is inactive.

    Returns None if either pair type is absent for this subject.
    """
    fm = feats[mask].float()
    ym = y_labels[mask]
    if fm.shape[0] == 0:
        return None
    model.eval()
    hidden_list = []
    with torch.no_grad():
        for s in range(0, len(fm), batch_size):
            hidden_list.append(model.get_hidden_state(fm[s:s + batch_size]).cpu().numpy())
    h = np.concatenate(hidden_list)          # (N, 64)
    imp_h = h[ym == 1]
    gen_h = h[ym == 0]
    if len(imp_h) == 0 or len(gen_h) == 0:
        return None
    return float(np.linalg.norm(imp_h.mean(axis=0) - gen_h.mean(axis=0)))


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
    te_pair_subjects_w2=None,
    batch_size=512,
    device='cpu',
    log=None,
    dropout: float = 0.0,
    # ── all_subjects mode ────────────────────────────────────────────────────
    all_subject_ids=None,
    combined_feats=None,
    combined_y=None,
    combined_subj1=None,
    combined_subj2=None,
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
    split_mode='all_subjects' → realistic threat model: attacker splits ALL subjects 50/50
                                 without knowing the real member/non-member division.
                                 Requires combined_feats/combined_y/combined_subj1/combined_subj2
                                 (the full pair pool: MM + NN + MN + NM concatenated) and
                                 all_subject_ids (members + non-members together).
                                 Training pairs: both sw1 and sw2 must be in shadow_in_k.
                                 Scoring: symmetric — subject X scored on all pairs where
                                 sw1==X OR sw2==X (both roles contribute to the delta).

    pair_subjects_w2 : array of shape (N,) giving the subject ID for window 2 of each training
                       pair (subj_win2_train from d5_attribution.npz). Used in member_aware mode
                       to prevent an OUT subject's gait from leaking into shadow training as x2.

    The three threat models compared in NB05c (all use split_mode='member_aware'):
      grey-box   : init_mode='target'    (real target model weights)
      BB warm    : init_mode='warm_base' (attacker's own surrogate weights)
      BB cold    : init_mode='random'    (random init)

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
        log.info('No checkpoint found, starting from scratch.')

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
    _pool_for_rng = all_subject_ids if split_mode == 'all_subjects' else member_list
    rng = np.random.default_rng(0)
    for _ in range(k_start):
        rng.choice(_pool_for_rng, int(len(_pool_for_rng) * INCL_RATE), replace=False)
        rng.random(len(combined_y) if split_mode == 'all_subjects' else len(y_tr))

    log.info(f'Training {K - k_start} shadow LSTMs (init={init_mode}, split={split_mode}, epochs={epochs})...')
    t0 = time.time()

    y_tr_tensor = torch.from_numpy(y_tr).long()
    if split_mode == 'all_subjects':
        _combined_y_tensor   = torch.from_numpy(combined_y).long()
        _combined_subj1      = combined_subj1
        _combined_subj2      = combined_subj2

    for k in range(k_start, K):
        if split_mode == 'all_subjects':
            # Realistic threat model: split ALL subjects without knowing member labels
            subject_draw = rng.choice(_pool_for_rng, int(len(_pool_for_rng) * INCL_RATE), replace=False)
            rng.random(len(combined_y))  # consume pair draw for RNG consistency

            shadow_in  = set(int(x) for x in subject_draw)
            shadow_out = set(int(x) for x in _pool_for_rng) - shadow_in

            # Training: only pairs where BOTH sw1 and sw2 are assigned to shadow_in
            included_pair_mask = (
                np.isin(_combined_subj1, list(shadow_in)) &
                np.isin(_combined_subj2, list(shadow_in))
            )
            loader = DataLoader(
                TensorDataset(
                    combined_feats[included_pair_mask].float(),
                    _combined_y_tensor[included_pair_mask],
                ),
                batch_size=batch_size, shuffle=True, num_workers=0,
            )
        elif split_mode == 'member_aware':
            subject_draw = rng.choice(member_list, int(len(member_list) * INCL_RATE), replace=False)
            pair_draw    = rng.random(len(y_tr))

            included           = set(int(x) for x in subject_draw)
            shadow_out         = set(member_list) - included
            included_pair_mask = np.isin(pair_subjects, list(included)) | (pair_subjects == -1)
            if pair_subjects_w2 is not None:
                included_pair_mask = included_pair_mask & ~np.isin(pair_subjects_w2, list(shadow_out))
            loader = DataLoader(
                TensorDataset(
                    all_feats[included_pair_mask].float(),
                    y_tr_tensor[included_pair_mask],
                ),
                batch_size=batch_size, shuffle=True, num_workers=0,
            )
        else:  # 'blind'
            subject_draw = rng.choice(member_list, int(len(member_list) * INCL_RATE), replace=False)
            pair_draw    = rng.random(len(y_tr))

            shadow_out         = set(int(x) for x in subject_draw)
            included_pair_mask = pair_draw < INCL_RATE
            loader = DataLoader(
                TensorDataset(
                    all_feats[included_pair_mask].float(),
                    y_tr_tensor[included_pair_mask],
                ),
                batch_size=batch_size, shuffle=True, num_workers=0,
            )

        shadow = ShadowLSTM(dropout=dropout).to(device)
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

        if split_mode == 'all_subjects':
            # Score every subject in shadow_out symmetrically (both sw1 and sw2 roles)
            for sid in shadow_out:
                impostor_sc, genuine_sc = compute_subject_scores_sym(
                    shadow, combined_feats, combined_y,
                    _combined_subj1, _combined_subj2, int(sid), batch_size
                )
                raw, lf, mf = all_deltas(impostor_sc, genuine_sc)
                if raw is not None:
                    out_raw[sid].append(raw)
                    out_lf[sid].append(lf)
                    out_mf[sid].append(mf)
        else:
            # OUT members: scored on train pairs (symmetric when pair_subjects_w2 available)
            for sid in shadow_out:
                sw1_mask = pair_subjects == sid
                if pair_subjects_w2 is not None:
                    sw1_mask = sw1_mask | (pair_subjects_w2 == sid)
                impostor_sc, genuine_sc = compute_subject_scores(
                    shadow, all_feats, y_tr, sw1_mask, batch_size
                )
                raw, lf, mf = all_deltas(impostor_sc, genuine_sc)
                if raw is not None:
                    out_raw[sid].append(raw)
                    out_lf[sid].append(lf)
                    out_mf[sid].append(mf)

            # Non-members: always OUT, scored on test pairs (symmetric when te_pair_subjects_w2 available)
            for sid in nonmember_list:
                te_sw1_mask = te_pair_subjects == sid
                if te_pair_subjects_w2 is not None:
                    te_sw1_mask = te_sw1_mask | (te_pair_subjects_w2 == sid)
                impostor_sc, genuine_sc = compute_subject_scores(
                    shadow, te_feats, y_te, te_sw1_mask, batch_size
                )
                raw, lf, mf = all_deltas(impostor_sc, genuine_sc)
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

    m_arr  = np.array([(sid, lira_scores[sid]) for sid in member_list    if sid in lira_scores], dtype=float).reshape(-1, 2)
    nm_arr = np.array([(sid, lira_scores[sid]) for sid in nonmember_list if sid in lira_scores], dtype=float).reshape(-1, 2)
    if len(m_arr) == 0 or len(nm_arr) == 0:
        # Not enough subjects with valid OUT estimates (K too small for min-count threshold)
        return (np.array([0., 1.]), np.array([0., 1.]), 0.5, 0.5, 0.5, lira_scores, m_arr, nm_arr)
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
