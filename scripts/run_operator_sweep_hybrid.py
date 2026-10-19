"""Hybrid operator sweep — 2048-CNN / 128-LSTM configuration.

Trains 6 operator LSTMs on the 2048-dim CNN backbone with a 128-dim broadcast
auth key, then measures MIA AUC under three conditions per seed:
  - correct key   (operator's own key)
  - original key  (seed 3000, wrong for the operator)
  - no key        (code cleared to None, identity bypass attempt)

Writes results to results/whuGAIT/cl8_key2048/08c_hyb_multiseed_results.json

Run from the gait_mia/ directory:
    python scripts/run_operator_sweep_hybrid.py
"""
import json, logging, sys, time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.cl_cnn    import CDMLGaitCNN
from src.models.auth_model import AuthModel

# ── Config ────────────────────────────────────────────────────────────────────
FM_CODE_DIM_TRAIN = 2048
FM_CODE_DIM_AUTH  = 128
ORIG_AUTH_SEED    = 3000
N_TASKS           = 4
SUBJECTS_PER_TASK = 24
N_CLASSES         = N_TASKS * SUBJECTS_PER_TASK
EPOCHS_LSTM       = 10
LR_LSTM           = 0.0025
WEIGHT_DECAY      = 0.0015
DROPOUT           = 0.3
BATCH_SIZE        = 512
SEED_BASE         = 1000
DEVICE            = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CKPT_DIR   = Path('checkpoints/whuGAIT/cl8_key2048')
RESULT_DIR = Path('results/whuGAIT/cl8_key2048')
RESULT_DIR.mkdir(parents=True, exist_ok=True)

log = logging.getLogger('hyb_sweep')
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler())
log.info(f'Device: {DEVICE}  |  CNN key: {FM_CODE_DIM_TRAIN}-dim  |  Auth key: {FM_CODE_DIM_AUTH}-dim')

# ── Load pairs ────────────────────────────────────────────────────────────────
log.info('Loading auth pairs...')
d = np.load(CKPT_DIR / 'auth_pairs_cl.npz')
X1_tr, X2_tr, y_tr, s1_tr = d['X1_tr'], d['X2_tr'], d['y_tr'], d['s1_tr']
X1_te, X2_te, y_te, s1_te = d['X1_te'], d['X2_te'], d['y_te'], d['s1_te']

criterion = nn.CrossEntropyLoss()

def pair_loader(X1, X2, y, shuffle=True):
    return DataLoader(
        TensorDataset(torch.from_numpy(X1).float(),
                      torch.from_numpy(X2).float(),
                      torch.from_numpy(y).long()),
        batch_size=BATCH_SIZE, shuffle=shuffle)

train_loader = pair_loader(X1_tr, X2_tr, y_tr, shuffle=True)
test_loader  = pair_loader(X1_te, X2_te, y_te, shuffle=False)

_orig_key = torch.from_numpy(
    np.random.default_rng(ORIG_AUTH_SEED).choice([-1., 1.], size=FM_CODE_DIM_AUTH).astype(np.float32))

# ── Helpers ───────────────────────────────────────────────────────────────────
def _load_backbone():
    bb = CDMLGaitCNN(N_CLASSES, fm_code_dim=FM_CODE_DIM_AUTH)
    for k in range(N_TASKS):
        bb.register_code(k, SEED_BASE + k)
    bb.load_state_dict(torch.load(CKPT_DIR / f'cdml_task{N_TASKS-1}.pt', map_location=DEVICE))
    return bb

def _train_op_auth(op_seed):
    op_key = torch.from_numpy(
        np.random.default_rng(op_seed).choice([-1., 1.], size=FM_CODE_DIM_AUTH).astype(np.float32))
    bb = _load_backbone()
    bb.set_fm_code_direct(op_key)
    auth = AuthModel(cnn_encoder=bb, dropout=DROPOUT).to(DEVICE)
    for p in auth.cnn.parameters():
        p.requires_grad = False
    opt = Adam(filter(lambda p: p.requires_grad, auth.parameters()),
               lr=LR_LSTM, weight_decay=WEIGHT_DECAY)
    best_acc, best_state = 0.0, None
    for ep in range(1, EPOCHS_LSTM + 1):
        auth.train()
        for x1, x2, yb in train_loader:
            x1, x2, yb = x1.to(DEVICE), x2.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); criterion(auth(x1, x2), yb).backward(); opt.step()
        auth.eval(); c = t = 0
        with torch.no_grad():
            for x1, x2, yb in test_loader:
                x1, x2, yb = x1.to(DEVICE), x2.to(DEVICE), yb.to(DEVICE)
                c += (auth(x1, x2).argmax(1) == yb).sum().item(); t += len(yb)
        acc = c / t
        if acc > best_acc:
            best_acc = acc; best_state = {k: v.clone() for k, v in auth.state_dict().items()}
        log.info(f'  ep {ep:2d}/{EPOCHS_LSTM}  acc={acc*100:.1f}%')
    auth.load_state_dict(best_state)
    return auth, best_acc, op_key

def _mia_auc(auth, key_tensor):
    if key_tensor is None:
        auth.cnn._active_code = None
    else:
        auth.cnn.set_fm_code_direct(key_tensor)
    md = {}; nd = {}
    for X1, X2, yp, s1, pool in [
        (X1_tr, X2_tr, y_tr, s1_tr, md),
        (X1_te, X2_te, y_te, s1_te, nd)
    ]:
        g = defaultdict(list); imp = defaultdict(list)
        for i in range(0, len(yp), BATCH_SIZE):
            x1b = torch.from_numpy(X1[i:i+BATCH_SIZE]).float().to(DEVICE)
            x2b = torch.from_numpy(X2[i:i+BATCH_SIZE]).float().to(DEVICE)
            with torch.no_grad():
                sc = torch.softmax(auth(x1b, x2b), 1)[:, 1].cpu().numpy()
            for j in range(len(sc)):
                s = s1[i + j]
                (g if yp[i+j] == 0 else imp)[s].append(sc[j])
        for s in set(s1):
            if s in g and s in imp:
                pool[s] = float(np.mean(imp[s])) - float(np.mean(g[s]))
    ms = list(md.values()); ns = list(nd.values())
    return float(roc_auc_score([1]*len(ms)+[0]*len(ns), ms+ns))

# ── Main sweep ────────────────────────────────────────────────────────────────
sweep_path = RESULT_DIR / '08c_hyb_multiseed_results.json'

if sweep_path.exists():
    log.info(f'Results already exist at {sweep_path}, skipping.')
    sys.exit(0)

op_seeds = [2000, 4000, 5000, 6000, 7000, 8000,
            9000, 10000, 11000, 12000, 13000, 14000]

# Load existing results so completed seeds are skipped
sweep_results = {}
if sweep_path.exists():
    with open(sweep_path) as f:
        sweep_results = json.load(f)
    done = sorted(int(k) for k in sweep_results)
    log.info(f'Resuming — already done: {done}')

t0 = time.time()

for op_s in op_seeds:
    if str(op_s) in sweep_results:
        log.info(f'[sweep] seed={op_s} already computed, skipping.')
        continue
    log.info(f'\n[sweep] === seed={op_s} ===')
    auth, acc, op_key = _train_op_auth(op_s)
    auc_correct  = _mia_auc(auth, op_key)
    auc_original = _mia_auc(auth, _orig_key)
    auc_nokey    = _mia_auc(auth, None)
    dot  = float((op_key * _orig_key).sum().item())
    sweep_results[str(op_s)] = {
        'auth_acc':      float(acc),
        'auc_correct':   auc_correct,
        'auc_original':  auc_original,
        'auc_no_key':    auc_nokey,
        'dot_to_orig':   dot,                      # in [-128, +128]; +128 = identical
        'cos_to_orig':   dot / FM_CODE_DIM_AUTH,   # in [-1, +1]
    }
    log.info(f'[sweep] seed={op_s}  acc={acc*100:.1f}%  '
             f'auc_c={auc_correct:.4f}  auc_o={auc_original:.4f}  auc_nk={auc_nokey:.4f}')
    # Save after each seed so partial results survive job cancellation
    with open(sweep_path, 'w') as f:
        json.dump(sweep_results, f, indent=2)

log.info(f'\n[sweep] Done in {time.time()-t0:.0f}s. Saved to {sweep_path}')
print('\nSeed     Auth      AUC correct   AUC original   AUC no-key   cos(op,orig)')
print('-' * 76)
for s in op_seeds:
    if str(s) not in sweep_results:
        continue
    r = sweep_results[str(s)]
    print(f"{s:<8} {r['auth_acc']*100:>7.2f}%  "
          f"{r['auc_correct']:>11.4f}  {r['auc_original']:>13.4f}  "
          f"{r['auc_no_key']:>10.4f}  {r['cos_to_orig']:>+12.3f}")
