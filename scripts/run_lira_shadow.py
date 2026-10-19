"""Standalone LiRA shadow-model training for any dataset.

Replicates the NB05b grey-box pipeline but runs outside Jupyter,
supports GPU, and resumes from an existing checkpoint.

Since the checkpoint stores k_done, running with K=64 when k_done=32
continues from where the previous run stopped — no wasted work.

Run from the gait_mia/ directory:
    python scripts/run_lira_shadow.py [--dataset whuGAIT] [--K 64] [--epochs 3]
"""
import sys, json, logging, argparse, time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config_loader  import dataset_dirs, dataset_files
from src.data.auth_dataset    import load_auth_dataset, normalize_auth
from src.data.pairs_loader    import load_auth_pairs, load_attribution
from src.models.gait_cnn      import GaitCNN
from src.models.auth_model    import AuthModel
from src.attacks.lira_shadow  import train_shadow_models, lira_eval, tpr_at_fpr

# ── Args ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='whuGAIT')
    p.add_argument('--K',       type=int, default=64,
                   help='Total number of shadow models (resumes from existing checkpoint)')
    p.add_argument('--epochs',  type=int, default=3,
                   help='Training epochs per shadow LSTM')
    p.add_argument('--batch',   type=int, default=512)
    p.add_argument('--dropout', type=float, default=0.3)
    return p.parse_args()

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger('lira_shadow')

def main():
    args = parse_args()
    DATASET = args.dataset
    K       = args.K
    EPOCHS  = args.epochs
    BATCH   = args.batch
    DROPOUT = args.dropout
    DEVICE  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    log.info(f'=== LiRA shadow training  dataset={DATASET} K={K} epochs={EPOCHS} device={DEVICE} ===')

    # ── Paths ─────────────────────────────────────────────────────────────────
    DATA_ROOT = Path('data')
    _dirs = dataset_dirs(
        Path('logs'), Path('checkpoints'),
        artifacts_base=Path('artifacts'),
        results_base=Path('results'),
        dataset=DATASET,
    )
    LOG_DIR      = _dirs['logs']
    ARTIFACT_DIR = _dirs['artifacts']
    CKPT_DIR     = _dirs['checkpoints']
    RESULT_DIR   = _dirs['results']
    PATHS        = dataset_files(LOG_DIR, CKPT_DIR, artifacts_dir=ARTIFACT_DIR)

    CKPT_TARGET = ARTIFACT_DIR / f'05b_lira_target_shadows_{DATASET}.json'

    # Add file log alongside the stream
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(LOG_DIR / f'lira_shadow_K{K}.log', mode='a')
    fh.setFormatter(logging.Formatter('%(asctime)s  %(message)s', datefmt='%H:%M:%S'))
    log.addHandler(fh)

    # ── Subject split ─────────────────────────────────────────────────────────
    with open(PATHS['split_json']) as f:
        split = json.load(f)
    member_list    = split['train_ids']
    nonmember_list = split['held_out_ids']
    N_CLASSES      = len(member_list)
    log.info(f'Dataset: {DATASET}  members={len(member_list)}  non-members={len(nonmember_list)}')

    # ── Check existing checkpoint ─────────────────────────────────────────────
    if CKPT_TARGET.exists():
        ckpt_data = json.load(open(CKPT_TARGET))
        k_done = ckpt_data.get('k_done', 0)
        log.info(f'Resuming from checkpoint: k_done={k_done}/{K}')
        if k_done >= K:
            log.info('Already complete — computing final AUC and exiting.')
            _report_auc(ckpt_data, member_list, nonmember_list, K)
            return
    else:
        log.info('No existing checkpoint — starting from scratch.')

    # ── Load CNN + target auth model ──────────────────────────────────────────
    norm_data = np.load(PATHS['norm_stats'])
    norm_mean, norm_std = norm_data['mean'], norm_data['std']

    cnn_n = json.load(open(PATHS['cnn_meta']))['n_classes'] if PATHS['cnn_meta'].exists() else N_CLASSES
    cnn_base = GaitCNN(n_classes=cnn_n)
    cnn_base.load_state_dict(torch.load(PATHS['cnn_ckpt'], map_location='cpu'))
    cnn_base.eval()

    target_auth = AuthModel(cnn_base)
    target_auth.load_state_dict(torch.load(PATHS['auth_ckpt'], map_location='cpu'))
    target_auth.eval()

    target_state = {
        'lstm': target_auth.lstm.state_dict(),
        'fc':   target_auth.fc.state_dict(),
    }

    # ── Load pairs + attribution ──────────────────────────────────────────────
    X1_tr, X2_tr, y_tr = load_auth_pairs('train', DATA_ROOT, ARTIFACT_DIR)
    X1_te, X2_te, y_te = load_auth_pairs('test',  DATA_ROOT, ARTIFACT_DIR)
    subj1_tr, subj2_tr = load_attribution('train', ARTIFACT_DIR)
    subj1_te, subj2_te = load_attribution('test',  ARTIFACT_DIR)

    try:
        X1_mn, X2_mn, y_mn = load_auth_pairs('mn', DATA_ROOT, ARTIFACT_DIR)
        X1_nm, X2_nm, y_nm = load_auth_pairs('nm', DATA_ROOT, ARTIFACT_DIR)
        subj1_mn, subj2_mn = load_attribution('mn', ARTIFACT_DIR)
        subj1_nm, subj2_nm = load_attribution('nm', ARTIFACT_DIR)
        has_cross = True
    except (KeyError, FileNotFoundError):
        log.info('MN/NM pairs not found — using train+test pool only')
        X1_mn = X2_mn = y_mn = np.array([])
        X1_nm = X2_nm = y_nm = np.array([])
        subj1_mn = subj2_mn = subj1_nm = subj2_nm = np.array([], dtype=np.int32)
        has_cross = False

    # ── Pre-compute CNN features (CPU — feature extraction is fast) ───────────
    def _norm(X1, X2):
        X1n, X2n, _ = normalize_auth(X1, X2, (norm_mean, norm_std))
        return X1n, X2n

    def _feats(X1n, X2n):
        parts = []
        with torch.no_grad():
            for s in range(0, len(X1n), BATCH):
                e = min(s + BATCH, len(X1n))
                f1 = cnn_base.get_feature_maps(torch.from_numpy(X1n[s:e]).float())
                f2 = cnn_base.get_feature_maps(torch.from_numpy(X2n[s:e]).float())
                parts.append(torch.cat([f1, f2], dim=1).half().cpu())
        return torch.cat(parts, dim=0)

    log.info('Pre-computing CNN features ...')
    t0 = time.time()
    feats_tr = _feats(*_norm(X1_tr, X2_tr))
    feats_te = _feats(*_norm(X1_te, X2_te))

    if has_cross:
        feats_mn = _feats(*_norm(X1_mn, X2_mn))
        feats_nm = _feats(*_norm(X1_nm, X2_nm))
        combined_feats  = torch.cat([feats_tr, feats_te, feats_mn, feats_nm], dim=0)
        combined_y      = np.concatenate([y_tr, y_te, y_mn, y_nm])
        combined_subj1  = np.concatenate([subj1_tr, subj1_te, subj1_mn, subj1_nm])
        combined_subj2  = np.concatenate([subj2_tr, subj2_te, subj2_mn, subj2_nm])
    else:
        combined_feats  = torch.cat([feats_tr, feats_te], dim=0)
        combined_y      = np.concatenate([y_tr, y_te])
        combined_subj1  = np.concatenate([subj1_tr, subj1_te])
        combined_subj2  = np.concatenate([subj2_tr, subj2_te])

    log.info(f'Features ready in {time.time()-t0:.0f}s  combined={combined_feats.shape[0]:,} pairs')

    # ── Train shadows ─────────────────────────────────────────────────────────
    log.info(f'Training {K} shadow models (device={DEVICE}) ...')
    t1 = time.time()
    out_raw, out_lf, out_mf = train_shadow_models(
        feats_tr, y_tr, subj1_tr, member_list,
        feats_te, y_te, subj1_te, nonmember_list,
        K=K, epochs=EPOCHS,
        init_mode='target',
        target_state=target_state,
        split_mode='all_subjects',
        checkpoint_path=CKPT_TARGET,
        batch_size=BATCH,
        device=str(DEVICE),
        log=log,
        dropout=DROPOUT,
        all_subject_ids=np.array(member_list + nonmember_list),
        combined_feats=combined_feats,
        combined_y=combined_y,
        combined_subj1=combined_subj1,
        combined_subj2=combined_subj2,
    )
    log.info(f'Shadow training done in {(time.time()-t1)/60:.1f} min')

    # ── Final AUC ─────────────────────────────────────────────────────────────
    ckpt_final = json.load(open(CKPT_TARGET))
    _report_auc(ckpt_final, member_list, nonmember_list, K)


def _report_auc(ckpt_data, member_list, nonmember_list, K):
    out_raw = {int(k): v for k, v in ckpt_data['raw'].items()}
    all_ids = member_list + nonmember_list
    valid   = [s for s in all_ids if s in out_raw and len(out_raw[s]) >= 5]
    scores  = [float(np.mean(out_raw[s])) for s in valid]
    labels  = [1 if s in set(member_list) else 0 for s in valid]

    if len(set(labels)) < 2:
        log.warning('Only one class in valid subjects — cannot compute AUC')
        return

    auc = roc_auc_score(labels, scores)
    log.info(f'')
    log.info(f'====== FINAL RESULTS  K={K} ======')
    log.info(f'  AUC:     {auc:.4f}')
    log.info(f'  Valid:   {len(valid)}/{len(all_ids)}  (mem={sum(labels)} nm={len(labels)-sum(labels)})')

    # Save summary
    summary = {
        'dataset':        ckpt_data.get('dataset', 'whuGAIT'),
        'K':              K,
        'auc':            auc,
        'valid_subjects': len(valid),
        'members':        sum(labels),
        'nonmembers':     len(labels) - sum(labels),
    }
    out_path = Path(f'results') / ckpt_data.get('dataset', 'whuGAIT') / f'05b_lira_K{K}_summary.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(out_path, 'w'), indent=2)
    log.info(f'  Summary: {out_path}')


if __name__ == '__main__':
    main()
