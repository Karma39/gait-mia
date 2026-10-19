"""
Compute Equal Error Rate (EER) for all experiments and write LaTeX macros.

Authentication EER  : from saved 03_auth_results.npz (all 4 datasets)
MIA EER             : from saved 04_mia_scores.npz (all 4 datasets)
Evasion EER         : epsilon budget at which ASR=50% (from 06b attack results)
CL Auth EER         : LSTM inference on auth_pairs_cl.npz (key128, hybrid)
CL MIA EER          : per-subject delta inference on CL LSTM (key128, hybrid)

Run from the gait_mia/ directory:
    python scripts/compute_eer.py
"""
import sys, json, logging
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_curve

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.cl_cnn    import CDMLGaitCNN
from src.models.auth_model import AuthModel

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger('compute_eer')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATASETS = ['whuGAIT', 'ucihar', 'wisdm', 'combined']

ARTIFACT_DIR   = Path('artifacts')
CKPT_DIR       = Path('checkpoints')
RESULT_DIR     = Path('results')
LATEX_DIR      = Path('latex/generated')
LATEX_DIR.mkdir(parents=True, exist_ok=True)

# ─── EER utilities ────────────────────────────────────────────────────────────

def compute_eer(y_true: np.ndarray, scores: np.ndarray) -> float:
    """EER from binary labels and continuous scores.

    scores: higher = more likely positive class (pos_label=1).
    EER is interpolated at the point where FAR = FRR.
    """
    fpr, tpr, _ = roc_curve(y_true, scores, pos_label=1)
    fnr = 1.0 - tpr
    # find crossing point by minimising |fpr - fnr|
    idx = np.argmin(np.abs(fpr - fnr))
    # linear interpolation between adjacent points
    if idx > 0 and idx < len(fpr) - 1:
        eer = float((fpr[idx] + fnr[idx]) / 2.0)
    else:
        eer = float((fpr[idx] + fnr[idx]) / 2.0)
    return eer


def mia_eer(member_scores: np.ndarray, nonmember_scores: np.ndarray) -> float:
    """EER for a binary MIA: member=1, non-member=0."""
    y = np.concatenate([np.ones(len(member_scores)),
                        np.zeros(len(nonmember_scores))])
    s = np.concatenate([member_scores, nonmember_scores])
    return compute_eer(y, s)


def evasion_eer_eps(pair_eps_min: np.ndarray,
                    pair_succeeded: np.ndarray) -> float:
    """Epsilon budget at which ASR crosses 50%.

    For resistant pairs we set eps_min = +inf. Returns the median budget.
    """
    eps = pair_eps_min.copy().astype(float)
    eps[~pair_succeeded] = np.inf
    finite_eps = eps[np.isfinite(eps)]
    if len(finite_eps) == 0:
        return float('nan')
    asr_50_eps = float(np.percentile(finite_eps, 50))
    return asr_50_eps


# ─── Authentication EER ───────────────────────────────────────────────────────

auth_eer = {}
for ds in DATASETS:
    path = ARTIFACT_DIR / ds / '03_auth_results.npz'
    if not path.exists():
        log.warning(f'[auth EER] {ds}: no artifact found, skipping')
        continue
    d = np.load(path)
    # Use the held-out test split (NN pairs: both subjects non-members)
    scores_nn = d['nn_scores']
    labels_nn = d['nn_labels']
    eer_nn = compute_eer(labels_nn, scores_nn)
    # Also compute on combined test (nn + mn + nm)
    scores_all = np.concatenate([d['nn_scores'], d['mn_scores'], d['nm_scores']])
    labels_all = np.concatenate([d['nn_labels'], d['mn_labels'], d['nm_labels']])
    eer_all = compute_eer(labels_all, scores_all)
    auth_eer[ds] = {'eer_nn': eer_nn, 'eer_all': eer_all}
    log.info(f'[auth EER] {ds}: NN={eer_nn:.4f}  all-test={eer_all:.4f}')


# ─── MIA EER ─────────────────────────────────────────────────────────────────

mia_eer_vals = {}
for ds in DATASETS:
    path = ARTIFACT_DIR / ds / '04_mia_scores.npz'
    if not path.exists():
        log.warning(f'[MIA EER] {ds}: no artifact found, skipping')
        continue
    d = np.load(path)
    eer = mia_eer(d['member_scores'], d['nonmember_scores'])
    mia_eer_vals[ds] = eer
    auc = float(d['mia_auc'])
    log.info(f'[MIA EER] {ds}: EER={eer:.4f}  AUC={auc:.4f}')


# ─── Hidden-layer delta EER ───────────────────────────────────────────────────

hidden_eer_vals = {}
for ds in DATASETS:
    path = ARTIFACT_DIR / ds / '05a_hidden_deltas.npz'
    if not path.exists():
        log.warning(f'[hidden EER] {ds}: no artifact found, skipping')
        continue
    d = np.load(path)
    eer = mia_eer(d['member_hidden_deltas'], d['nonmember_hidden_deltas'])
    hidden_eer_vals[ds] = eer
    auc = float(d['auc_hidden'])
    log.info(f'[hidden EER] {ds}: EER={eer:.4f}  AUC={auc:.4f}')


# ─── Evasion EER (epsilon at ASR=50%) ─────────────────────────────────────────

evasion_eer_vals = {}
for ds in DATASETS:
    path = ARTIFACT_DIR / ds / f'06b_{ds}_attack_results.npz'
    if not path.exists():
        log.warning(f'[evasion EER] {ds}: no artifact found, skipping')
        continue
    d = np.load(path, allow_pickle=True)
    eps50 = evasion_eer_eps(d['pair_eps_min'], d['pair_succeeded'])
    asr_total = float(d['pair_succeeded'].mean())
    evasion_eer_vals[ds] = eps50
    log.info(f'[evasion EER] {ds}: eps@ASR50={eps50:.4f}  overall ASR={asr_total:.4f}')


# ─── CL Authentication EER ────────────────────────────────────────────────────

def run_cl_auth_inference(auth_model_path, pairs_npz, backbone_cfg):
    """Load LSTM auth, run inference on test pairs, return (scores, labels)."""
    N_CLASSES = 4 * 24
    N_TASKS   = 4
    SEED_BASE = 1000

    fm_code_dim_train = backbone_cfg['fm_code_dim_train']
    fm_code_dim_auth  = backbone_cfg['fm_code_dim_auth']
    auth_seed         = backbone_cfg['auth_seed']
    ckpt_dir          = backbone_cfg['ckpt_dir']

    rng = np.random.default_rng(auth_seed)
    auth_fm_code = torch.from_numpy(
        rng.choice([-1.0, 1.0], size=fm_code_dim_auth).astype(np.float32)
    )

    backbone = CDMLGaitCNN(N_CLASSES, fm_code_dim=fm_code_dim_train)
    backbone.load_state_dict(
        torch.load(ckpt_dir / f'cdml_task{N_TASKS-1}.pt', map_location=DEVICE)
    )
    backbone.code_dim = fm_code_dim_auth
    backbone.set_fm_code_direct(auth_fm_code)
    backbone.eval()

    auth = AuthModel(backbone)
    auth.load_state_dict(torch.load(auth_model_path, map_location=DEVICE))
    auth.eval()

    d = np.load(pairs_npz)
    X1 = torch.from_numpy(d['X1_te'].astype(np.float32))
    X2 = torch.from_numpy(d['X2_te'].astype(np.float32))
    y  = d['y_te'].astype(np.int32)

    scores_list = []
    batch = 1024
    with torch.no_grad():
        for i in range(0, len(X1), batch):
            logits = auth(X1[i:i+batch].to(DEVICE), X2[i:i+batch].to(DEVICE))
            probs  = torch.softmax(logits, dim=1)[:, 1]
            scores_list.append(probs.cpu().numpy())
    scores = np.concatenate(scores_list)
    return scores, y


def run_cl_mia_inference(auth_model, pairs_npz):
    """Compute per-subject delta scores for member and non-member pools.

    delta(s) = mean P(different|impostor pairs) - mean P(different|genuine pairs)
    """
    d = np.load(pairs_npz)
    results = {}
    for pool, sfx in [('member', 'tr'), ('nonmember', 'te')]:
        X1 = torch.from_numpy(d[f'X1_{sfx}'].astype(np.float32))
        X2 = torch.from_numpy(d[f'X2_{sfx}'].astype(np.float32))
        y  = d[f'y_{sfx}'].astype(np.int32)
        s1 = d[f's1_{sfx}'].astype(np.int32)

        scores_list = []
        batch = 1024
        with torch.no_grad():
            for i in range(0, len(X1), batch):
                logits = auth_model(
                    X1[i:i+batch].to(DEVICE), X2[i:i+batch].to(DEVICE)
                )
                probs = torch.softmax(logits, dim=1)[:, 1]
                scores_list.append(probs.cpu().numpy())
        scores = np.concatenate(scores_list)

        # Per-subject delta: mean impostor score - mean genuine score
        subj_delta = {}
        for subj in np.unique(s1):
            mask = s1 == subj
            imp_mask  = mask & (y == 1)
            gen_mask  = mask & (y == 0)
            if imp_mask.sum() > 0 and gen_mask.sum() > 0:
                delta = scores[imp_mask].mean() - scores[gen_mask].mean()
                subj_delta[subj] = float(delta)
        results[pool] = np.array(list(subj_delta.values()), dtype=np.float32)
    return results


# CL key128
log.info('\n=== CL key128 ===')
cl128_cfg = {
    'fm_code_dim_train': 128,
    'fm_code_dim_auth':  128,
    'auth_seed':         3000,
    'ckpt_dir':          CKPT_DIR / 'whuGAIT/cl8_key128',
}
pairs_cl128 = CKPT_DIR / 'whuGAIT/cl8_key128/auth_pairs_cl.npz'
auth_ckpt128 = CKPT_DIR / 'whuGAIT/cl8_key128/auth_cdml_fm.pt'

cl_auth_eer = {}
cl_mia_eer  = {}

if auth_ckpt128.exists() and pairs_cl128.exists():
    scores128, labels128 = run_cl_auth_inference(auth_ckpt128, pairs_cl128, cl128_cfg)
    eer128 = compute_eer(labels128, scores128)
    cl_auth_eer['key128'] = eer128
    log.info(f'[CL auth EER] key128: {eer128:.4f}')

    # CL MIA EER
    N_CLASSES = 4 * 24
    backbone128 = CDMLGaitCNN(N_CLASSES, fm_code_dim=128)
    backbone128.load_state_dict(
        torch.load(CKPT_DIR / 'whuGAIT/cl8_key128/cdml_task3.pt', map_location=DEVICE)
    )
    rng = np.random.default_rng(3000)
    code128 = torch.from_numpy(rng.choice([-1.0, 1.0], size=128).astype(np.float32))
    backbone128.set_fm_code_direct(code128)
    backbone128.eval()
    auth128 = AuthModel(backbone128)
    auth128.load_state_dict(torch.load(auth_ckpt128, map_location=DEVICE))
    auth128.eval()
    deltas128 = run_cl_mia_inference(auth128, pairs_cl128)
    eer_mia128 = mia_eer(deltas128['member'], deltas128['nonmember'])
    cl_mia_eer['key128'] = eer_mia128
    log.info(f'[CL MIA EER] key128: {eer_mia128:.4f}  '
             f'(member_mean={deltas128["member"].mean():.3f}  '
             f'nm_mean={deltas128["nonmember"].mean():.3f})')


# CL hybrid (2048-CNN, 128-LSTM)
log.info('\n=== CL hybrid (2048-CNN / 128-LSTM) ===')
cl_hyb_cfg = {
    'fm_code_dim_train': 2048,
    'fm_code_dim_auth':  128,
    'auth_seed':         3000,
    'ckpt_dir':          CKPT_DIR / 'whuGAIT/cl8_key2048',
}
pairs_clhyb  = CKPT_DIR / 'whuGAIT/cl8_key2048/auth_pairs_cl.npz'
auth_ckpt_hyb = CKPT_DIR / 'whuGAIT/cl8_key2048/auth_cdml_fm_128authkey.pt'

if auth_ckpt_hyb.exists() and pairs_clhyb.exists():
    scores_hyb, labels_hyb = run_cl_auth_inference(auth_ckpt_hyb, pairs_clhyb, cl_hyb_cfg)
    eer_hyb = compute_eer(labels_hyb, scores_hyb)
    cl_auth_eer['hybrid'] = eer_hyb
    log.info(f'[CL auth EER] hybrid: {eer_hyb:.4f}')

    # CL MIA EER (hybrid)
    backbone_hyb = CDMLGaitCNN(N_CLASSES, fm_code_dim=2048)
    backbone_hyb.load_state_dict(
        torch.load(CKPT_DIR / 'whuGAIT/cl8_key2048/cdml_task3.pt', map_location=DEVICE)
    )
    backbone_hyb.code_dim = 128
    rng = np.random.default_rng(3000)
    code_hyb = torch.from_numpy(rng.choice([-1.0, 1.0], size=128).astype(np.float32))
    backbone_hyb.set_fm_code_direct(code_hyb)
    backbone_hyb.eval()
    auth_hyb = AuthModel(backbone_hyb)
    auth_hyb.load_state_dict(torch.load(auth_ckpt_hyb, map_location=DEVICE))
    auth_hyb.eval()
    deltas_hyb = run_cl_mia_inference(auth_hyb, pairs_clhyb)
    eer_mia_hyb = mia_eer(deltas_hyb['member'], deltas_hyb['nonmember'])
    cl_mia_eer['hybrid'] = eer_mia_hyb
    log.info(f'[CL MIA EER] hybrid: {eer_mia_hyb:.4f}  '
             f'(member_mean={deltas_hyb["member"].mean():.3f}  '
             f'nm_mean={deltas_hyb["nonmember"].mean():.3f})')


# ─── Save results to JSON ──────────────────────────────────────────────────────

eer_results = {
    'auth_eer':     auth_eer,
    'mia_eer':      mia_eer_vals,
    'evasion_eer':  evasion_eer_vals,
    'cl_auth_eer':  cl_auth_eer,
    'cl_mia_eer':   cl_mia_eer,
}
out_json = RESULT_DIR / 'eer_results.json'
RESULT_DIR.mkdir(parents=True, exist_ok=True)
with open(out_json, 'w') as f:
    json.dump(eer_results, f, indent=2)
log.info(f'\nSaved EER results to {out_json}')


# ─── Write LaTeX macros ────────────────────────────────────────────────────────

def pct(v: float) -> str:
    return f'{v * 100:.2f}\\%'


lines = [
    '% Auto-generated by compute_eer.py: DO NOT EDIT MANUALLY',
    '% Equal Error Rate macros for all experiments',
    '',
]

# Authentication EER per dataset
ds_label = {'whuGAIT': 'WhuGait', 'ucihar': 'Ucihar', 'wisdm': 'Wisdm', 'combined': 'Combined'}
for ds, label in ds_label.items():
    v = auth_eer.get(ds, {})
    nn = v.get('eer_nn', float('nan'))
    al = v.get('eer_all', float('nan'))
    lines += [
        f'\\providecommand{{\\eer{label}Auth}}{{}}',
        f'\\renewcommand{{\\eer{label}Auth}}{{{pct(nn)}}}',
        f'\\providecommand{{\\eer{label}AuthAll}}{{}}',
        f'\\renewcommand{{\\eer{label}AuthAll}}{{{pct(al)}}}',
    ]

lines.append('')

# MIA EER per dataset
for ds, label in ds_label.items():
    v = mia_eer_vals.get(ds, float('nan'))
    lines += [
        f'\\providecommand{{\\eer{label}MIA}}{{}}',
        f'\\renewcommand{{\\eer{label}MIA}}{{{pct(v)}}}',
    ]

lines.append('')

# Hidden-layer delta EER per dataset
for ds, label in ds_label.items():
    v = hidden_eer_vals.get(ds, float('nan'))
    lines += [
        f'\\providecommand{{\\eer{label}HiddenDelta}}{{}}',
        f'\\renewcommand{{\\eer{label}HiddenDelta}}{{{pct(v)}}}',
    ]

lines.append('')

# Evasion EER (epsilon at ASR=50%)
for ds, label in ds_label.items():
    v = evasion_eer_vals.get(ds, float('nan'))
    lines += [
        f'\\providecommand{{\\eerEvasion{label}EpsHalf}}{{}}',
        f'\\renewcommand{{\\eerEvasion{label}EpsHalf}}{{{v:.3f}}}',
    ]

lines.append('')

# CL Auth EER
for cond, macro in [('key128', 'CLKeyAuth'), ('hybrid', 'CLHybAuth')]:
    v = cl_auth_eer.get(cond, float('nan'))
    lines += [
        f'\\providecommand{{\\eer{macro}}}{{}}',
        f'\\renewcommand{{\\eer{macro}}}{{{pct(v)}}}',
    ]

lines.append('')

# CL MIA EER
for cond, macro in [('key128', 'CLKeyMIA'), ('hybrid', 'CLHybMIA')]:
    v = cl_mia_eer.get(cond, float('nan'))
    lines += [
        f'\\providecommand{{\\eer{macro}}}{{}}',
        f'\\renewcommand{{\\eer{macro}}}{{{pct(v)}}}',
    ]

latex_out = LATEX_DIR / 'eer_metrics.tex'
with open(latex_out, 'w') as f:
    f.write('\n'.join(lines) + '\n')
log.info(f'Wrote LaTeX macros to {latex_out}')

# ─── Summary ──────────────────────────────────────────────────────────────────

log.info('\n=== EER Summary ===')
for ds in DATASETS:
    a  = auth_eer.get(ds, {}).get('eer_nn', float('nan'))
    m  = mia_eer_vals.get(ds, float('nan'))
    ev = evasion_eer_vals.get(ds, float('nan'))
    log.info(f'{ds:12s}  auth EER={a*100:.2f}%  MIA EER={m*100:.2f}%  '
             f'evasion eps@50%={ev:.3f}')

log.info('')
for cond in ['key128', 'hybrid']:
    ca = cl_auth_eer.get(cond, float('nan'))
    cm = cl_mia_eer.get(cond, float('nan'))
    log.info(f'CL {cond:7s}  auth EER={ca*100:.2f}%  MIA EER={cm*100:.2f}%')
