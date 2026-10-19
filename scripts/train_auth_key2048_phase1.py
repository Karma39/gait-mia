"""Phase-1 auth run for 2048-dim key: trains auth LSTM + three-condition MIA only.
No operator sweep — run this first to compare auth accuracy and MIA gap vs 128-key.

Run from the gait_mia/ directory:
    python scripts/train_auth_key2048_phase1.py
"""
import sys, json, time, logging
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
FM_CODE_DIM   = 2048
AUTH_FM_SEED  = 3000
WRONG_KEY_SEED = 9999
N_TASKS       = 4
SUBJECTS_PER_TASK = 24
N_CLASSES     = N_TASKS * SUBJECTS_PER_TASK
EPOCHS_LSTM   = 25          # more than 128-key (10) to compensate for slower convergence
LR_LSTM       = 0.0025
WEIGHT_DECAY  = 0.0015
DROPOUT       = 0.3
BATCH_SIZE    = 512
SEED_BASE     = 1000
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CKPT_DIR   = Path('checkpoints/whuGAIT/cl8_key2048')
RESULT_DIR = Path('results/whuGAIT/cl8_key2048')

log = logging.getLogger('phase1_key2048')
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler())
log.info(f'Device: {DEVICE}  |  FM_CODE_DIM: {FM_CODE_DIM}  |  EPOCHS: {EPOCHS_LSTM}')

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

train_loader = pair_loader(X1_tr, X2_tr, y_tr)
test_loader  = pair_loader(X1_te, X2_te, y_te, shuffle=False)
log.info(f'Train pairs: {len(y_tr)}  Test pairs: {len(y_te)}')

# ── Load auth key ─────────────────────────────────────────────────────────────
with open(CKPT_DIR / 'auth_fm_code.json') as f:
    _fm_data = json.load(f)
auth_fm_code = torch.tensor(_fm_data['code'], dtype=torch.float32)
wrong_key    = torch.from_numpy(
    np.random.default_rng(WRONG_KEY_SEED).choice([-1., 1.], size=FM_CODE_DIM).astype(np.float32))
log.info(f'Auth key: seed={_fm_data["seed"]}  dim={len(_fm_data["code"])}')

# ── Build backbone ────────────────────────────────────────────────────────────
backbone = CDMLGaitCNN(N_CLASSES, fm_code_dim=FM_CODE_DIM)
for k in range(N_TASKS):
    backbone.register_code(k, SEED_BASE + k)
backbone.load_state_dict(
    torch.load(CKPT_DIR / f'cdml_task{N_TASKS-1}.pt', map_location=DEVICE))
backbone.set_fm_code_direct(auth_fm_code)

auth      = AuthModel(cnn_encoder=backbone, dropout=DROPOUT).to(DEVICE)
auth_ckpt = CKPT_DIR / 'auth_cdml_fm.pt'
for p in auth.cnn.parameters():
    p.requires_grad = False

# ── Train or load ─────────────────────────────────────────────────────────────
if auth_ckpt.exists():
    log.info('Checkpoint found, loading...')
    auth.load_state_dict(torch.load(auth_ckpt, map_location=DEVICE))
    best_acc = None
else:
    opt = Adam(filter(lambda p: p.requires_grad, auth.parameters()),
               lr=LR_LSTM, weight_decay=WEIGHT_DECAY)
    best_acc, best_state = 0.0, None
    t0 = time.time()
    for ep in range(1, EPOCHS_LSTM + 1):
        auth.train()
        for x1, x2, yb in train_loader:
            x1, x2, yb = x1.to(DEVICE), x2.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); criterion(auth(x1, x2), yb).backward(); opt.step()
        auth.eval()
        tr_c = tr_t = te_c = te_t = 0
        with torch.no_grad():
            for x1, x2, yb in train_loader:
                x1, x2, yb = x1.to(DEVICE), x2.to(DEVICE), yb.to(DEVICE)
                tr_c += (auth(x1, x2).argmax(1) == yb).sum().item(); tr_t += len(yb)
            for x1, x2, yb in test_loader:
                x1, x2, yb = x1.to(DEVICE), x2.to(DEVICE), yb.to(DEVICE)
                te_c += (auth(x1, x2).argmax(1) == yb).sum().item(); te_t += len(yb)
        acc = te_c / te_t
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in auth.state_dict().items()}
        log.info(f'ep {ep:2d}/{EPOCHS_LSTM}  train={tr_c/tr_t*100:.2f}%  test={acc*100:.2f}%  best={best_acc*100:.2f}%')
    auth.load_state_dict(best_state)
    torch.save(auth.state_dict(), auth_ckpt)
    log.info(f'Training done in {time.time()-t0:.0f}s  best={best_acc*100:.2f}%')

# ── Three-condition MIA ───────────────────────────────────────────────────────
def mia_auc(model, key):
    model.eval()
    model.cnn.set_fm_code_direct(key)
    md = {}; nd = {}
    for X1, X2, yp, s1, pool in [(X1_tr,X2_tr,y_tr,s1_tr,md),(X1_te,X2_te,y_te,s1_te,nd)]:
        g = defaultdict(list); imp = defaultdict(list)
        for i in range(0, len(yp), BATCH_SIZE):
            x1b = torch.from_numpy(X1[i:i+BATCH_SIZE]).float()
            x2b = torch.from_numpy(X2[i:i+BATCH_SIZE]).float()
            with torch.no_grad():
                sc = torch.softmax(model(x1b, x2b), 1)[:, 1].numpy()
            for j in range(len(sc)):
                s = s1[i+j]
                (g if yp[i+j]==0 else imp)[s].append(sc[j])
        for s in set(s1):
            if s in g and s in imp:
                pool[s] = float(np.mean(imp[s])) - float(np.mean(g[s]))
    ms = list(md.values()); ns = list(nd.values())
    return float(roc_auc_score([1]*len(ms)+[0]*len(ns), ms+ns))

log.info('\n=== Three-condition MIA ===')

# no key
auth.cnn.clear_task()
auc_no_key = mia_auc(auth, auth_fm_code)   # dummy — we just need clear_task active
auth.cnn.clear_task()
md_nk = {}; nd_nk = {}
auth.eval()
for X1, X2, yp, s1, pool in [(X1_tr,X2_tr,y_tr,s1_tr,md_nk),(X1_te,X2_te,y_te,s1_te,nd_nk)]:
    g = defaultdict(list); imp = defaultdict(list)
    for i in range(0, len(yp), BATCH_SIZE):
        x1b = torch.from_numpy(X1[i:i+BATCH_SIZE]).float()
        x2b = torch.from_numpy(X2[i:i+BATCH_SIZE]).float()
        with torch.no_grad():
            sc = torch.softmax(auth(x1b, x2b), 1)[:, 1].numpy()
        for j in range(len(sc)):
            s = s1[i+j]
            (g if yp[i+j]==0 else imp)[s].append(sc[j])
    for s in set(s1):
        if s in g and s in imp:
            pool[s] = float(np.mean(imp[s])) - float(np.mean(g[s]))
ms_nk = list(md_nk.values()); ns_nk = list(nd_nk.values())
auc_no_key = float(roc_auc_score([1]*len(ms_nk)+[0]*len(ns_nk), ms_nk+ns_nk))

auth.cnn.set_fm_code_direct(auth_fm_code)
auc_correct = mia_auc(auth, auth_fm_code)
auc_wrong   = mia_auc(auth, wrong_key)

results = {
    'fm_code_dim':   FM_CODE_DIM,
    'auth_acc':      float(best_acc) if best_acc else None,
    'auc_no_key':    auc_no_key,
    'auc_correct':   auc_correct,
    'auc_wrong':     auc_wrong,
    'delta_correct': auc_correct - auc_no_key,
    'delta_wrong':   auc_wrong   - auc_no_key,
}
with open(RESULT_DIR / 'mia_phase1_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print('\n' + '='*50)
print(f'KEY SIZE: {FM_CODE_DIM}-dim  (vs 128-key baseline)')
print('='*50)
print(f'Auth accuracy:   {best_acc*100:.2f}%   (128-key: 75.41%)')  if best_acc else print('Auth accuracy:   (loaded from ckpt)')
print(f'MIA no key:      {auc_no_key:.4f}   (128-key: 0.9151)')
print(f'MIA correct key: {auc_correct:.4f}   (128-key: 0.8937)  Δ={auc_correct-auc_no_key:+.4f}')
print(f'MIA wrong key:   {auc_wrong:.4f}   (128-key: 0.5620)  Δ={auc_wrong-auc_no_key:+.4f}')
print(f'AUC gap:         {(auc_no_key - auc_wrong)*100:.1f} pp   (128-key: 35.3 pp)')
print(f'\nSaved to {RESULT_DIR}/mia_phase1_results.json')
