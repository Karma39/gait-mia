"""CL training with 2048-dim CDML key (16×128, per-channel-per-feature).

Mirrors NB08a but targets checkpoints/whuGAIT/cl8_key2048/.
Run from the gait_mia/ directory:
    python scripts/train_cl_key2048.py
"""
import sys, json, time, logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.gait_cnn import GaitCNN
from src.models.cl_cnn   import CDMLGaitCNN
from src.data.dataset    import load_dataset, filter_subjects, normalize

# ── Config ────────────────────────────────────────────────────────────────────
FM_CODE_DIM       = 2048           # 16 channels × 128 features
AUTH_FM_SEED      = 3000
N_TASKS           = 4
SUBJECTS_PER_TASK = 24
N_CLASSES         = N_TASKS * SUBJECTS_PER_TASK
EPOCHS_PER_TASK   = 100
LR, LR_GAMMA      = 1e-3, 0.9
BATCH_SIZE        = 512
SEED_BASE         = 1000
DEVICE            = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_ROOT    = Path('data')
ARTIFACT_DIR = Path('artifacts/whuGAIT')
CKPT_DIR     = Path('checkpoints/whuGAIT/cl8_key2048')
RESULT_DIR   = Path('results/whuGAIT/cl8_key2048')
CKPT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

log = logging.getLogger('train_key2048')
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler())
log.info(f'Device: {DEVICE}  |  FM_CODE_DIM: {FM_CODE_DIM}')

# ── Data ──────────────────────────────────────────────────────────────────────
with open(ARTIFACT_DIR / 'subject_split.json') as f:
    split = json.load(f)

all_member_ids = sorted(split['train_ids'])
member_ids     = all_member_ids[:N_TASKS * SUBJECTS_PER_TASK]
id_to_cls      = {sid: i for i, sid in enumerate(member_ids)}
tasks          = [{'task_id': k,
                   'subject_ids': member_ids[k*SUBJECTS_PER_TASK:(k+1)*SUBJECTS_PER_TASK]}
                  for k in range(N_TASKS)]

X_raw_tr, y_raw_tr = load_dataset(str(DATA_ROOT / 'Dataset #1'), 'train')
X_raw_te, y_raw_te = load_dataset(str(DATA_ROOT / 'Dataset #1'), 'test')
X_mem_tr, _        = filter_subjects(X_raw_tr, y_raw_tr, member_ids)
_, norm_stats       = normalize(X_mem_tr)
X_tr, _ = normalize(X_raw_tr, norm_stats)
X_te, _ = normalize(X_raw_te, norm_stats)

task_window_counts = [np.isin(y_raw_tr, t['subject_ids']).sum() for t in tasks]
min_windows        = min(task_window_counts)
log.info(f'Windows per task: {task_window_counts}  →  subsample to {min_windows}')

def make_loader_equal(X, y, sids, cls_map, n_max, seed=42):
    mask = np.isin(y, sids)
    Xs, ys = X[mask], y[mask]
    ys = np.array([cls_map[s] for s in ys], dtype=np.int64)
    if len(Xs) > n_max:
        idx = np.random.default_rng(seed).choice(len(Xs), n_max, replace=False)
        Xs, ys = Xs[idx], ys[idx]
    return DataLoader(TensorDataset(torch.from_numpy(Xs).float(),
                                    torch.from_numpy(ys).long()),
                      batch_size=BATCH_SIZE, shuffle=True)

def make_loader(X, y, sids, cls_map):
    mask = np.isin(y, sids)
    Xs, ys = X[mask], y[mask]
    ys = np.array([cls_map[s] for s in ys], dtype=np.int64)
    return DataLoader(TensorDataset(torch.from_numpy(Xs).float(),
                                    torch.from_numpy(ys).long()),
                      batch_size=BATCH_SIZE, shuffle=False)

task_train = [make_loader_equal(X_tr, y_raw_tr, t['subject_ids'], id_to_cls,
                                min_windows, seed=SEED_BASE+t['task_id']) for t in tasks]
task_eval  = [make_loader(X_te, y_raw_te, t['subject_ids'], id_to_cls) for t in tasks]

# ── Training ──────────────────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()

def accuracy(model, loader, task_id=None):
    if task_id is not None and hasattr(model, 'set_task'):
        model.set_task(task_id)
    model.eval()
    corr = tot = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            corr += (model(x).argmax(1) == y).sum().item()
            tot  += len(y)
    return corr / tot

# Only train CDML (std baseline already exists from cl8)
label = 'cdml'
if all((CKPT_DIR / f'{label}_task{k}.pt').exists() for k in range(N_TASKS)):
    log.info('All checkpoints found, loading without retraining.')
    model = CDMLGaitCNN(N_CLASSES, fm_code_dim=FM_CODE_DIM).to(DEVICE)
    for k in range(N_TASKS):
        model.register_code(k, SEED_BASE + k)
    per_task_acc = []
    for tid in range(N_TASKS):
        model.load_state_dict(torch.load(CKPT_DIR / f'{label}_task{tid}.pt', map_location=DEVICE))
        for k in range(N_TASKS):
            model.register_code(k, SEED_BASE + k)
        pt = {i: accuracy(model, task_eval[i], task_id=i) for i in range(tid+1)}
        per_task_acc.append(pt)
else:
    model = CDMLGaitCNN(N_CLASSES, fm_code_dim=FM_CODE_DIM).to(DEVICE)
    per_task_acc = []
    t_total = time.time()

    for t in tasks:
        tid = t['task_id']
        model.register_code(tid, SEED_BASE + tid)
        model.set_task(tid)
        optim = Adam(model.parameters(), lr=LR)
        sched = ExponentialLR(optim, LR_GAMMA)
        t0 = time.time()
        for ep in range(1, EPOCHS_PER_TASK + 1):
            model.train()
            for x, y in task_train[tid]:
                x, y = x.to(DEVICE), y.to(DEVICE)
                optim.zero_grad()
                criterion(model(x), y).backward()
                optim.step()
            sched.step()
            if ep % 10 == 0:
                acc_now = accuracy(model, task_eval[tid], task_id=tid)
                log.info(f'  [task {tid}] ep {ep:3d}/{EPOCHS_PER_TASK}  acc={acc_now*100:.1f}%')

        pt = {i: accuracy(model, task_eval[i], task_id=i) for i in range(tid+1)}
        per_task_acc.append(pt)
        torch.save(model.state_dict(), CKPT_DIR / f'{label}_task{tid}.pt')
        log.info(f'[task {tid}] done in {time.time()-t0:.0f}s  '
                 + '  '.join(f'T{i}={pt[i]*100:.1f}%' for i in range(tid+1)))

    log.info(f'Total training time: {time.time()-t_total:.0f}s')

# ── Accuracy matrix ───────────────────────────────────────────────────────────
print(f'\nCDML (key2048) per-task accuracy:')
print('          ' + ''.join(f'   T{j}  ' for j in range(N_TASKS)))
for i, pt in enumerate(per_task_acc):
    row = f'After T{i}:  '
    for j in range(N_TASKS):
        row += f'{pt[j]*100:5.1f}%  ' if j in pt else '   --   '
    print(row)

acc_out = {str(i): {str(j): float(v) for j, v in pt.items()}
           for i, pt in enumerate(per_task_acc)}
with open(RESULT_DIR / 'cl_acc_matrix.json', 'w') as f:
    json.dump(acc_out, f, indent=2)
log.info(f'Saved accuracy matrix to {RESULT_DIR}/cl_acc_matrix.json')

# ── Save auth key (2048-dim) ──────────────────────────────────────────────────
auth_key_np = np.random.default_rng(AUTH_FM_SEED).choice(
    [-1.0, 1.0], size=FM_CODE_DIM).astype(np.float32)
auth_key_path = CKPT_DIR / 'auth_fm_code.json'
with open(auth_key_path, 'w') as f:
    json.dump({'seed': AUTH_FM_SEED, 'dim': FM_CODE_DIM,
               'code': auth_key_np.tolist()}, f, indent=2)
log.info(f'Saved auth key: {auth_key_path}  (dim={FM_CODE_DIM}, seed={AUTH_FM_SEED})')
print('Done.')
