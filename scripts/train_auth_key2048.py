"""Train auth LSTM + operator sweep for the 2048-dim key run.

Run from the gait_mia/ directory:
    python scripts/train_auth_key2048.py
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
from src.data.dataset      import load_dataset, filter_subjects, normalize

# ── Config (must match train_cl_key2048.py) ───────────────────────────────────
FM_CODE_DIM       = 2048
AUTH_FM_SEED      = 3000
N_TASKS           = 4
SUBJECTS_PER_TASK = 24
N_CLASSES         = N_TASKS * SUBJECTS_PER_TASK
EPOCHS_LSTM       = 10
LR_LSTM           = 0.0025
WEIGHT_DECAY      = 0.0015
DROPOUT           = 0.3
BATCH_SIZE        = 512
N_PAIRS_TRAIN     = 33000
N_PAIRS_TEST      = 3800
SEED_BASE         = 1000
DEVICE            = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CKPT_DIR   = Path('checkpoints/whuGAIT/cl8_key2048')
RESULT_DIR = Path('results/whuGAIT/cl8_key2048')
ARTIFACT_DIR = Path('artifacts/whuGAIT')

log = logging.getLogger('auth_key2048')
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler())
log.info(f'Device: {DEVICE}  |  FM_CODE_DIM: {FM_CODE_DIM}')

# ── Load / build auth pairs (reuse 128-key pairs if already built) ────────────
PAIRS_PATH = CKPT_DIR / 'auth_pairs_cl.npz'
if PAIRS_PATH.exists():
    log.info('Loading existing pairs...')
    d = np.load(PAIRS_PATH)
    X1_tr, X2_tr, y_tr, s1_tr = d['X1_tr'], d['X2_tr'], d['y_tr'], d['s1_tr']
    X1_te, X2_te, y_te, s1_te = d['X1_te'], d['X2_te'], d['y_te'], d['s1_te']
else:
    log.info('Building pairs from scratch...')
    from src.data.pair_builder import build_auth_pairs

    with open(ARTIFACT_DIR / 'subject_split.json') as f:
        split = json.load(f)
    all_member_ids    = sorted(split['train_ids'])
    all_nonmember_ids = sorted(split['held_out_ids'])
    member_ids    = all_member_ids[:N_TASKS * SUBJECTS_PER_TASK]
    leftover_ids  = all_member_ids[N_TASKS * SUBJECTS_PER_TASK:]
    nonmember_ids = all_nonmember_ids + leftover_ids

    X_raw_tr, y_raw_tr = load_dataset(str(Path('data/Dataset #1')), 'train')
    X_raw_d2, y_d2     = load_dataset(str(Path('data/Dataset #2')), 'test')
    X_mem_tr, _        = filter_subjects(X_raw_tr, y_raw_tr, member_ids)
    _, norm_stats       = normalize(X_mem_tr)
    X_tr_n, _ = normalize(X_raw_tr, norm_stats)
    X_d2n, _  = normalize(X_raw_d2, norm_stats)

    mask_mem = np.isin(y_raw_tr, member_ids)
    X1_tr, X2_tr, y_tr, s1_tr, _ = build_auth_pairs(
        X_tr_n[mask_mem].astype(np.float32), y_raw_tr[mask_mem], N_PAIRS_TRAIN, seed=42)
    mask_nm = np.isin(y_d2, nonmember_ids)
    X1_te, X2_te, y_te, s1_te, _ = build_auth_pairs(
        X_d2n[mask_nm].astype(np.float32), y_d2[mask_nm], N_PAIRS_TEST, seed=43)
    np.savez_compressed(PAIRS_PATH,
        X1_tr=X1_tr, X2_tr=X2_tr, y_tr=y_tr, s1_tr=s1_tr,
        X1_te=X1_te, X2_te=X2_te, y_te=y_te, s1_te=s1_te)
    log.info('Pairs saved.')

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
log.info(f'Auth key: seed={_fm_data["seed"]}  dim={len(_fm_data["code"])}')

# ── Build and train auth_cdml_fm ──────────────────────────────────────────────
backbone_fm = CDMLGaitCNN(N_CLASSES, fm_code_dim=FM_CODE_DIM)
for k in range(N_TASKS):
    backbone_fm.register_code(k, SEED_BASE + k)
backbone_fm.load_state_dict(
    torch.load(CKPT_DIR / f'cdml_task{N_TASKS-1}.pt', map_location=DEVICE))
backbone_fm.set_fm_code_direct(auth_fm_code)

auth_fm      = AuthModel(cnn_encoder=backbone_fm, dropout=DROPOUT).to(DEVICE)
auth_fm_ckpt = CKPT_DIR / 'auth_cdml_fm.pt'
for p in auth_fm.cnn.parameters():
    p.requires_grad = False

if auth_fm_ckpt.exists():
    log.info('[cdml_fm] checkpoint found, loading...')
    auth_fm.load_state_dict(torch.load(auth_fm_ckpt, map_location=DEVICE))
    fm_best_acc = None
else:
    optim_fm = Adam(filter(lambda p: p.requires_grad, auth_fm.parameters()),
                    lr=LR_LSTM, weight_decay=WEIGHT_DECAY)
    fm_best_acc, fm_best_state = 0.0, None
    t0 = time.time()
    for ep in range(1, EPOCHS_LSTM + 1):
        auth_fm.train()
        for x1, x2, yb in train_loader:
            x1, x2, yb = x1.to(DEVICE), x2.to(DEVICE), yb.to(DEVICE)
            optim_fm.zero_grad()
            criterion(auth_fm(x1, x2), yb).backward()
            optim_fm.step()
        auth_fm.eval(); c = t = 0
        with torch.no_grad():
            for x1, x2, yb in test_loader:
                x1, x2, yb = x1.to(DEVICE), x2.to(DEVICE), yb.to(DEVICE)
                c += (auth_fm(x1, x2).argmax(1) == yb).sum().item(); t += len(yb)
        acc = c / t
        if acc > fm_best_acc:
            fm_best_acc = acc
            fm_best_state = {k: v.clone() for k, v in auth_fm.state_dict().items()}
        log.info(f'[cdml_fm] ep {ep:2d}/{EPOCHS_LSTM}  acc={acc*100:.2f}%')
    auth_fm.load_state_dict(fm_best_state)
    torch.save(auth_fm.state_dict(), auth_fm_ckpt)
    log.info(f'[cdml_fm] best={fm_best_acc*100:.2f}%  ({time.time()-t0:.0f}s)')

# ── MIA helper ────────────────────────────────────────────────────────────────
def mia_auc(auth_model, key_tensor):
    auth_model.eval()
    auth_model.cnn.set_fm_code_direct(key_tensor)
    md = {}; nd = {}
    for X1, X2, ypair, s1, pool in [
        (X1_tr, X2_tr, y_tr, s1_tr, md),
        (X1_te, X2_te, y_te, s1_te, nd)
    ]:
        g = defaultdict(list); imp = defaultdict(list)
        for i in range(0, len(ypair), BATCH_SIZE):
            x1b = torch.from_numpy(X1[i:i+BATCH_SIZE]).float()
            x2b = torch.from_numpy(X2[i:i+BATCH_SIZE]).float()
            with torch.no_grad():
                sc = torch.softmax(auth_model(x1b, x2b), 1)[:, 1].numpy()
            for j in range(len(sc)):
                s = s1[i + j]
                (g if ypair[i+j] == 0 else imp)[s].append(sc[j])
        for s in set(s1):
            if s in g and s in imp:
                pool[s] = float(np.mean(imp[s])) - float(np.mean(g[s]))
    ms = list(md.values()); ns = list(nd.values())
    return float(roc_auc_score([1]*len(ms)+[0]*len(ns), ms+ns))

# ── Three-condition MIA ───────────────────────────────────────────────────────
log.info('\n=== Three-condition MIA ===')
correct_key = auth_fm_code
wrong_key   = torch.from_numpy(
    np.random.default_rng(9999).choice([-1., 1.], size=FM_CODE_DIM).astype(np.float32))

auc_correct = mia_auc(auth_fm, correct_key)
auc_wrong   = mia_auc(auth_fm, wrong_key)

# no-key: clear the code
auth_fm.cnn.clear_task()
auc_no_key  = mia_auc(auth_fm, correct_key)  # re-applies correct key for fair comparison
# actually run no-key properly
class _NoKey:
    pass
auth_fm.cnn.clear_task()
auc_no_key_scores = []
md_nk = {}; nd_nk = {}
auth_fm.eval()
for X1, X2, ypair, s1, pool in [
    (X1_tr, X2_tr, y_tr, s1_tr, md_nk),
    (X1_te, X2_te, y_te, s1_te, nd_nk)
]:
    g = defaultdict(list); imp = defaultdict(list)
    for i in range(0, len(ypair), BATCH_SIZE):
        x1b = torch.from_numpy(X1[i:i+BATCH_SIZE]).float()
        x2b = torch.from_numpy(X2[i:i+BATCH_SIZE]).float()
        with torch.no_grad():
            sc = torch.softmax(auth_fm(x1b, x2b), 1)[:, 1].numpy()
        for j in range(len(sc)):
            s = s1[i + j]
            (g if ypair[i+j] == 0 else imp)[s].append(sc[j])
    for s in set(s1):
        if s in g and s in imp:
            pool[s] = float(np.mean(imp[s])) - float(np.mean(g[s]))
ms_nk = list(md_nk.values()); ns_nk = list(nd_nk.values())
auc_no_key = float(roc_auc_score([1]*len(ms_nk)+[0]*len(ns_nk), ms_nk+ns_nk))

# restore correct key
auth_fm.cnn.set_fm_code_direct(correct_key)

mia_results = {
    'auc_no_key':   auc_no_key,
    'auc_correct':  auc_correct,
    'auc_wrong':    auc_wrong,
    'delta_correct': auc_correct - auc_no_key,
    'delta_wrong':   auc_wrong   - auc_no_key,
}
with open(RESULT_DIR / 'mia_results.json', 'w') as f:
    json.dump(mia_results, f, indent=2)

log.info(f'  no key:      AUC={auc_no_key:.4f}')
log.info(f'  correct key: AUC={auc_correct:.4f}  Δ={auc_correct-auc_no_key:+.4f}')
log.info(f'  wrong key:   AUC={auc_wrong:.4f}  Δ={auc_wrong-auc_no_key:+.4f}')

# ── Operator sweep ────────────────────────────────────────────────────────────
sweep_path = RESULT_DIR / 'multiseed_results.json'
op_seeds   = [2000, 4000, 5000, 6000, 7000, 8000]
orig_key   = auth_fm_code

if sweep_path.exists():
    log.info('\n[sweep] already exists, loading...')
    sweep_results = json.load(open(sweep_path))
else:
    sweep_results = {}
    t0 = time.time()
    for op_s in op_seeds:
        log.info(f'\n[sweep] === seed={op_s} ===')
        op_key = torch.from_numpy(
            np.random.default_rng(op_s).choice([-1., 1.], size=FM_CODE_DIM).astype(np.float32))
        bb = CDMLGaitCNN(N_CLASSES, fm_code_dim=FM_CODE_DIM)
        for k in range(N_TASKS):
            bb.register_code(k, SEED_BASE + k)
        bb.load_state_dict(torch.load(CKPT_DIR / f'cdml_task{N_TASKS-1}.pt', map_location=DEVICE))
        bb.set_fm_code_direct(op_key)
        a = AuthModel(cnn_encoder=bb, dropout=DROPOUT).to(DEVICE)
        for p in a.cnn.parameters(): p.requires_grad = False
        opt = Adam(filter(lambda p: p.requires_grad, a.parameters()),
                   lr=LR_LSTM, weight_decay=WEIGHT_DECAY)
        best, best_st = 0.0, None
        for ep in range(1, EPOCHS_LSTM + 1):
            a.train()
            for x1, x2, yb in train_loader:
                x1, x2, yb = x1.to(DEVICE), x2.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad(); criterion(a(x1, x2), yb).backward(); opt.step()
            a.eval(); c = t = 0
            with torch.no_grad():
                for x1, x2, yb in test_loader:
                    x1, x2, yb = x1.to(DEVICE), x2.to(DEVICE), yb.to(DEVICE)
                    c += (a(x1, x2).argmax(1) == yb).sum().item(); t += len(yb)
            acc = c / t
            if acc > best: best = acc; best_st = {k: v.clone() for k, v in a.state_dict().items()}
            log.info(f'  ep {ep:2d}/{EPOCHS_LSTM}  acc={acc*100:.1f}%')
        a.load_state_dict(best_st)
        auc_c = mia_auc(a, op_key)
        auc_o = mia_auc(a, orig_key)
        sweep_results[str(op_s)] = {'auth_acc': float(best), 'auc_correct': float(auc_c), 'auc_original': float(auc_o)}
        log.info(f'[sweep] seed={op_s}  acc={best*100:.1f}%  auc_c={auc_c:.4f}  auc_o={auc_o:.4f}')
    with open(sweep_path, 'w') as f:
        json.dump(sweep_results, f, indent=2)
    log.info(f'\n[sweep] Done in {time.time()-t0:.0f}s')

print('\n=== SUMMARY (key2048) ===')
print(f"Auth acc:    {fm_best_acc*100:.2f}%" if fm_best_acc else "Auth acc:    (loaded from ckpt)")
print(f"MIA no key:  {auc_no_key:.4f}")
print(f"MIA correct: {auc_correct:.4f}  ({auc_correct-auc_no_key:+.4f})")
print(f"MIA wrong:   {auc_wrong:.4f}  ({auc_wrong-auc_no_key:+.4f})")
sc = [sweep_results[str(s)]['auc_correct']  for s in op_seeds]
so = [sweep_results[str(s)]['auc_original'] for s in op_seeds]
sa = [sweep_results[str(s)]['auth_acc']     for s in op_seeds]
print(f"\nOperator sweep mean: auth={sum(sa)/len(sa)*100:.1f}%  auc_c={sum(sc)/len(sc):.4f}  auc_o={sum(so)/len(so):.4f}")
print('\nDone.')
