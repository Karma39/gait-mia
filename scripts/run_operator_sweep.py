"""Standalone operator sweep — trains 6 operator LSTMs and writes 08c_multiseed_results.json.

Run from the gait_mia/ directory:
    python scripts/run_operator_sweep.py
"""
import json, logging, sys, time
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.cl_cnn import CDMLGaitCNN
from src.models.auth_model import AuthModel

# ── Config (must match NB08b) ─────────────────────────────────────────────────
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

CKPT_DIR   = Path('checkpoints/whuGAIT/cl8')
RESULT_DIR = Path('results/whuGAIT')

log = logging.getLogger('sweep')
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler())

# ── Load pairs ────────────────────────────────────────────────────────────────
log.info('Loading auth pairs...')
d = np.load(CKPT_DIR / 'auth_pairs_cl.npz')
X1_tr, X2_tr, y_tr, s1_tr = d['X1_tr'], d['X2_tr'], d['y_tr'], d['s1_tr']
X1_te, X2_te, y_te, s1_te = d['X1_te'], d['X2_te'], d['y_te'], d['s1_te']

import torch.nn as nn
criterion = nn.CrossEntropyLoss()

def pair_loader(X1, X2, y, shuffle=True):
    return DataLoader(
        TensorDataset(torch.from_numpy(X1).float(),
                      torch.from_numpy(X2).float(),
                      torch.from_numpy(y).long()),
        batch_size=BATCH_SIZE, shuffle=shuffle)

train_loader = pair_loader(X1_tr, X2_tr, y_tr, shuffle=True)
test_loader  = pair_loader(X1_te, X2_te, y_te, shuffle=False)

_orig_key = torch.tensor(
    json.load(open(CKPT_DIR / 'auth_fm_code.json'))['code'], dtype=torch.float32)

# ── Helpers ───────────────────────────────────────────────────────────────────
def _train_op_auth(op_seed):
    _op_key = torch.from_numpy(
        np.random.default_rng(op_seed).choice([-1., 1.], size=128).astype(np.float32))
    _bb = CDMLGaitCNN(N_CLASSES)
    for _k in range(N_TASKS):
        _bb.register_code(_k, SEED_BASE + _k)
    _bb.load_state_dict(torch.load(CKPT_DIR / f'cdml_task{N_TASKS-1}.pt', map_location=DEVICE))
    _bb.set_fm_code_direct(_op_key)
    _a = AuthModel(cnn_encoder=_bb, dropout=DROPOUT).to(DEVICE)
    for p in _a.cnn.parameters():
        p.requires_grad = False
    _opt = Adam(filter(lambda p: p.requires_grad, _a.parameters()),
                lr=LR_LSTM, weight_decay=WEIGHT_DECAY)
    _best, _best_st = 0.0, None
    for _ep in range(1, EPOCHS_LSTM + 1):
        _a.train()
        for _x1, _x2, _yb in train_loader:
            _x1, _x2, _yb = _x1.to(DEVICE), _x2.to(DEVICE), _yb.to(DEVICE)
            _opt.zero_grad(); criterion(_a(_x1, _x2), _yb).backward(); _opt.step()
        _a.eval(); _c = _t = 0
        with torch.no_grad():
            for _x1, _x2, _yb in test_loader:
                _x1, _x2, _yb = _x1.to(DEVICE), _x2.to(DEVICE), _yb.to(DEVICE)
                _c += (_a(_x1, _x2).argmax(1) == _yb).sum().item(); _t += len(_yb)
        _acc = _c / _t
        if _acc > _best:
            _best = _acc; _best_st = {k: v.clone() for k, v in _a.state_dict().items()}
        log.info(f'  ep {_ep:2d}/{EPOCHS_LSTM}  acc={_acc*100:.1f}%')
    _a.load_state_dict(_best_st)
    return _a, _best, _op_key

def _mia_auc(auth_model, key_tensor):
    from collections import defaultdict as _dd
    from sklearn.metrics import roc_auc_score as _ras
    auth_model.cnn.set_fm_code_direct(key_tensor)
    _md = {}; _nd = {}
    for _X1, _X2, _ypair, _s1, _pool in [
        (X1_tr, X2_tr, y_tr, s1_tr, _md),
        (X1_te, X2_te, y_te, s1_te, _nd)
    ]:
        _g = _dd(list); _imp = _dd(list)
        for _i in range(0, len(_ypair), BATCH_SIZE):
            _x1b = torch.from_numpy(_X1[_i:_i+BATCH_SIZE]).float()
            _x2b = torch.from_numpy(_X2[_i:_i+BATCH_SIZE]).float()
            with torch.no_grad():
                _sc = torch.softmax(auth_model(_x1b, _x2b), 1)[:, 1].numpy()
            for _j in range(len(_sc)):
                _s = _s1[_i + _j]
                (_g if _ypair[_i+_j] == 0 else _imp)[_s].append(_sc[_j])
        for _s in set(_s1):
            if _s in _g and _s in _imp:
                _pool[_s] = float(np.mean(_imp[_s])) - float(np.mean(_g[_s]))
    _ms = list(_md.values()); _ns = list(_nd.values())
    return float(_ras([1]*len(_ms)+[0]*len(_ns), _ms+_ns))

# ── Main sweep ────────────────────────────────────────────────────────────────
sweep_path = RESULT_DIR / '08c_multiseed_results.json'

if sweep_path.exists():
    log.info(f'Results already exist at {sweep_path}, skipping.')
    sys.exit(0)

op_seeds = [2000, 4000, 5000, 6000, 7000, 8000]
sweep_results = {}
t0 = time.time()

for op_s in op_seeds:
    log.info(f'\n[sweep] === seed={op_s} ===')
    op_auth, op_acc, op_key = _train_op_auth(op_s)
    auc_c = _mia_auc(op_auth, op_key)
    auc_o = _mia_auc(op_auth, _orig_key)
    sweep_results[str(op_s)] = {
        'auth_acc': float(op_acc),
        'auc_correct': float(auc_c),
        'auc_original': float(auc_o),
    }
    log.info(f'[sweep] seed={op_s}  acc={op_acc*100:.1f}%  auc_c={auc_c:.4f}  auc_o={auc_o:.4f}')

with open(sweep_path, 'w') as f:
    json.dump(sweep_results, f, indent=2)

log.info(f'\n[sweep] Done in {time.time()-t0:.0f}s. Saved to {sweep_path}')
print('\nSeed     Auth      AUC correct   AUC original')
print('-' * 48)
for s in op_seeds:
    r = sweep_results[str(s)]
    print(f"{s:<8} {r['auth_acc']*100:>7.2f}%  {r['auc_correct']:>11.4f}  {r['auc_original']:>12.4f}")
