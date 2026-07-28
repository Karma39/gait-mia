import yaml
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.parent


def load_config(config_path: str | Path | None = None) -> dict:
    """Load config.yaml from the repo root (or a custom path)."""
    path = Path(config_path) if config_path else _REPO_ROOT / 'config.yaml'
    with open(path) as f:
        return yaml.safe_load(f)


def get_dataset(config_path: str | Path | None = None) -> str:
    """Shorthand: return the active dataset name."""
    return load_config(config_path)['dataset']


def get_encoder_dataset(config_path: str | Path | None = None) -> str | None:
    """Return the encoder_dataset override, or None if within-dataset mode."""
    return load_config(config_path).get('encoder_dataset') or None


def dataset_dirs(
    logs_base: str | Path,
    ckpt_base: str | Path,
    artifacts_base: str | Path | None = None,
    emb_base: str | Path | None = None,
    results_base: str | Path | None = None,
    dataset: str | None = None,
) -> dict:
    """Return per-dataset subdirectory paths and create them.

    Each dataset gets its own subdirectory under every base directory so
    switching config.yaml never overwrites another dataset's artifacts.

    Directory roles:
        logs/        : text .log files only (human-readable execution logs)
        artifacts/   : structured data files (.npz, .json) read by analysis notebooks
        checkpoints/ : model weights (.pt files)
        results/     : figures and plots

    Returns a dict with keys:
        logs, checkpoints, artifacts (if artifacts_base given),
        embeddings_cnn (if emb_base given), results (if results_base given)
    """
    ds = dataset or get_dataset()
    dirs = {
        'logs':        Path(logs_base) / ds,
        'checkpoints': Path(ckpt_base) / ds,
    }
    if artifacts_base is not None:
        dirs['artifacts'] = Path(artifacts_base) / ds
    if emb_base is not None:
        dirs['embeddings_cnn'] = Path(emb_base) / ds / 'cnn'
    if results_base is not None:
        dirs['results'] = Path(results_base) / ds
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def dataset_files(logs_dir=None, ckpt_dir=None, artifacts_dir=None,
                  dataset: str | None = None) -> dict:
    """Return paths to all dataset-specific files.

    logs_dir       : dataset-specific logs subdirectory (e.g. logs/whuGAIT/)
    artifacts_dir  : dataset-specific artifacts subdirectory (e.g. artifacts/whuGAIT/)
    ckpt_dir       : dataset-specific checkpoints subdirectory

    If artifacts_dir is None, falls back to logs_dir for backward compatibility.

    Keys always present:
        split_json   : subject_split.json
        norm_stats   : auth_norm_stats.npz

    Keys present when ckpt_dir is given:
        cnn_ckpt     : cnn_encoder.pt
        cnn_meta     : cnn_encoder_meta.json
        auth_ckpt    : auth_model.pt
    """
    data_dir = Path(artifacts_dir) if artifacts_dir is not None else Path(logs_dir)
    paths = {
        'split_json': data_dir / 'subject_split.json',
        'norm_stats': data_dir / 'auth_norm_stats.npz',
    }
    if ckpt_dir is not None:
        paths['cnn_ckpt']  = Path(ckpt_dir) / 'cnn_encoder.pt'
        paths['cnn_meta']  = Path(ckpt_dir) / 'cnn_encoder_meta.json'
        paths['auth_ckpt'] = Path(ckpt_dir) / 'auth_model.pt'
    return paths
