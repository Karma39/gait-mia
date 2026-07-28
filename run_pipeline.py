#!/usr/bin/env python3
"""run_pipeline.py: execute the full thesis analysis pipeline.

Requires: nbformat, jupyter_client, ipykernel, pyyaml  (all in the gait env)
Run with: conda run -n gait python run_pipeline.py [run_names...]

Usage:
    python run_pipeline.py                   # all configured runs
    python run_pipeline.py whuGAIT           # single dataset
    python run_pipeline.py whuGAIT ucihar    # specific datasets
    python run_pipeline.py --quick whuGAIT   # smoke-test (few epochs, small samples)
    python run_pipeline.py --list            # show available runs
    python run_pipeline.py --dry-run         # show what would run
    python run_pipeline.py --from 05b_mia_attack.ipynb whuGAIT  # resume mid-pipeline
    python run_pipeline.py --until 03_train_authenticator.ipynb whuGAIT

Executed notebooks are saved to executed/{run_name}/ with all outputs embedded.
Source notebooks are never modified.
"""

import sys, json, re, time, traceback
from pathlib import Path

import yaml
import nbformat
import jupyter_client

# ── Directory layout ─────────────────────────────────────────────────────────
REPO_ROOT     = Path(__file__).parent
NOTEBOOKS_DIR = REPO_ROOT / 'notebooks'
EXECUTED_DIR  = REPO_ROOT / 'executed'

# ── Run configurations ────────────────────────────────────────────────────────
PIPELINE_CORE = [
    '02_train_cnn_encoder.ipynb',
    '03_train_authenticator.ipynb',
    '04_mia_signal.ipynb',
    '05a_delta_analysis.ipynb',
    '05b_mia_attack.ipynb',
    '05c_shadow_comparison.ipynb',
]

EVASION = [
    '06a_attack_setup.ipynb',
    '06b_attack.ipynb',
    '06c_attack_validation.ipynb',
]

EXPLORE = {
    'whuGAIT':  ['01_explore_datasets.ipynb'],
    'ucihar':   ['01_explore_ucihar.ipynb'],
    'wisdm':    ['01_explore_wisdm.ipynb'],
    'combined': ['01_explore_combined.ipynb'],
}

RUNS = {
    # ── Within-dataset ───────────────────────────────────────────────────────
    'whuGAIT': {
        'dataset':         'whuGAIT',
        'encoder_dataset': None,
        'notebooks':       EXPLORE['whuGAIT'] + PIPELINE_CORE + EVASION,
    },
    'ucihar': {
        'dataset':         'ucihar',
        'encoder_dataset': None,
        'notebooks':       EXPLORE['ucihar'] + PIPELINE_CORE + EVASION,
    },
    'wisdm': {
        'dataset':         'wisdm',
        'encoder_dataset': None,
        'notebooks':       EXPLORE['wisdm'] + PIPELINE_CORE + EVASION,
    },
    # ── Combined cross-dataset evaluation ───────────────────────────────────────
    # Prerequisite: 'whuGAIT' run must be complete (CNN checkpoint reused).
    'combined': {
        'dataset':         'combined',
        'encoder_dataset': 'whuGAIT',
        'notebooks':       EXPLORE['combined'] + PIPELINE_CORE + EVASION,
    },
}

# ── Config loading ────────────────────────────────────────────────────────────

def load_run_config(quick: bool = False) -> dict:
    """Load config.yaml and return a flat overrides dict for parameter injection.

    In quick mode, values from the `quick:` block override the main blocks.
    """
    cfg_path = REPO_ROOT / 'config.yaml'
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    cnn   = cfg.get('cnn', {})
    auth  = cfg.get('auth', {})
    mia   = cfg.get('mia', {})
    lira  = cfg.get('lira', {})
    pgd   = cfg.get('pgd', {})
    q     = cfg.get('quick', {}) if quick else {}

    return {
        # CNN_EPOCHS aliases to EPOCHS inside the notebook (NB02)
        'CNN_EPOCHS':    q.get('cnn_epochs',  cnn.get('epochs', 25)),
        # AUTH_EPOCHS aliases to EPOCHS inside the notebook (NB03)
        'AUTH_EPOCHS':   q.get('auth_epochs', auth.get('epochs', 10)),
        'AUTH_LR':       auth.get('lr', 0.0025),
        'AUTH_DROPOUT':  auth.get('dropout', 0.3),
        # LiRA shadow models (NB05b, NB05c)
        'K_LIRA':           q.get('lira_k_shadows',        lira.get('k_shadows', 32)),
        'LIRA_EPOCHS':      q.get('lira_epochs',           lira.get('epochs', 3)),
        'SURROGATE_EPOCHS': q.get('lira_surrogate_epochs', lira.get('surrogate_epochs', 5)),
        # PGD evasion (NB06a, NB06b)
        'K_PGD':              q.get('pgd_k_pgd',                 pgd.get('k_pgd', 20)),
        'N_BISECT':           q.get('pgd_n_bisect',               pgd.get('n_bisect', 6)),
        'N_COMBOS_TR_SAMPLE': q.get('pgd_n_combos_train_sample', pgd.get('n_combos_train_sample', 500)),
        'SEED_TR_SAMPLE':     pgd.get('seed_train_sample', 42),
    }

# ── Parameter injection ───────────────────────────────────────────────────────

_INJECT_MARKER = '# <<< RUNNER INJECTS THIS'

def inject_params(
    nb: nbformat.NotebookNode,
    dataset: str,
    encoder_dataset,
    overrides: dict | None = None,
) -> nbformat.NotebookNode:
    """Patch injected parameters in notebook cells.

    Two injection mechanisms:
      1. DATASET / ENCODER_DATASET: found in the first cell containing
         'DATASET' and the marker or get_dataset().
      2. All other overrides: any cell line ending with  # <<< RUNNER INJECTS THIS
         whose left-hand variable name is a key in `overrides`.
    """
    overrides = overrides or {}

    # Build full overrides including dataset vars
    all_overrides: dict[str, str] = {
        'DATASET':         repr(dataset),
        'ENCODER_DATASET': repr(encoder_dataset),
    }
    for k, v in overrides.items():
        all_overrides[k] = repr(v)

    for cell in nb.cells:
        if cell.cell_type != 'code':
            continue
        src = cell.source
        if _INJECT_MARKER not in src and 'get_dataset()' not in src:
            continue

        lines = src.split('\n')
        new_lines = []
        changed = False
        for line in lines:
            stripped = line.lstrip()
            # Check if it's a variable assignment with the marker
            m = re.match(r'([A-Z_][A-Z0-9_]*)\s*=', stripped)
            if m and (line.rstrip().endswith(_INJECT_MARKER) or 'get_dataset()' in line):
                var = m.group(1)
                if var in all_overrides:
                    indent = len(line) - len(stripped)
                    new_lines.append(f"{line[:indent]}{var} = {all_overrides[var]}  {_INJECT_MARKER}")
                    changed = True
                    continue
            new_lines.append(line)
        if changed:
            cell.source = '\n'.join(new_lines)
    return nb

# ── Helpers ───────────────────────────────────────────────────────────────────

def _ts() -> str:
    return time.strftime('%H:%M:%S')

def _cell_label(src: str, max_len: int = 72) -> str:
    """Return the first meaningful line of a cell as a short label."""
    for line in src.split('\n'):
        s = line.strip()
        if s and not s.startswith('#') and not s.startswith('%'):
            return s[:max_len] + ('…' if len(s) > max_len else '')
    return src.split('\n')[0][:max_len]

# ── Kernel-based cell execution ───────────────────────────────────────────────

def execute_notebook(
    nb: nbformat.NotebookNode,
    nb_dir: Path,
    timeout: int = 14400,
    kernel_name: str = 'python3',
) -> nbformat.NotebookNode:
    """Execute all cells using jupyter_client. Streams output live to stdout."""
    km = jupyter_client.KernelManager(kernel_name=kernel_name)
    km.start_kernel()
    kc = km.client()
    kc.start_channels()
    kc.wait_for_ready(timeout=60)

    # Set working directory to notebooks/ so relative paths (../data/, ../logs/) resolve
    kc.execute(f"import os; os.chdir({str(nb_dir)!r})")
    while True:
        msg = kc.get_iopub_msg(timeout=30)
        if (msg['header']['msg_type'] == 'status'
                and msg['content'].get('execution_state') == 'idle'):
            break

    n_code = sum(
        1 for c in nb.cells
        if c.cell_type == 'code' and c.source.strip()
    )
    cell_num = 0

    try:
        for i, cell in enumerate(nb.cells):
            if cell.cell_type != 'code':
                continue
            src = cell.source.strip()
            if not src:
                continue

            cell_num += 1
            label = _cell_label(src)
            print(f"    [{_ts()}] cell {cell_num:2d}/{n_code}  {label}", flush=True)
            t_cell = time.time()

            kc.execute(src)
            outputs = []
            exec_count = None

            while True:
                try:
                    msg = kc.get_iopub_msg(timeout=timeout)
                except Exception:
                    raise TimeoutError(
                        f"Cell {cell_num} silent for >{timeout}s, kernel may be hung"
                    )

                msg_type = msg['header']['msg_type']
                content  = msg['content']

                if msg_type == 'execute_input':
                    exec_count = content.get('execution_count')

                elif msg_type == 'stream':
                    text = content['text']
                    # Indent so it reads as nested under the cell header
                    for line in text.rstrip('\n').split('\n'):
                        print(f"             {line}", flush=True)
                    outputs.append(nbformat.v4.new_output(
                        output_type='stream',
                        name=content['name'],
                        text=text,
                    ))

                elif msg_type == 'display_data':
                    outputs.append(nbformat.v4.new_output(
                        output_type='display_data',
                        data=content['data'],
                        metadata=content.get('metadata', {}),
                    ))

                elif msg_type == 'execute_result':
                    outputs.append(nbformat.v4.new_output(
                        output_type='execute_result',
                        data=content['data'],
                        metadata=content.get('metadata', {}),
                        execution_count=content.get('execution_count'),
                    ))

                elif msg_type == 'error':
                    outputs.append(nbformat.v4.new_output(
                        output_type='error',
                        ename=content['ename'],
                        evalue=content['evalue'],
                        traceback=content['traceback'],
                    ))
                    cell.outputs = outputs
                    if exec_count is not None:
                        cell.execution_count = exec_count
                    raise RuntimeError(
                        f"Cell {cell_num} raised {content['ename']}: {content['evalue']}"
                    )

                elif msg_type == 'status' and content.get('execution_state') == 'idle':
                    elapsed_cell = time.time() - t_cell
                    print(f"    [{_ts()}] cell {cell_num:2d}/{n_code}  ✓ {elapsed_cell:.1f}s",
                          flush=True)
                    break

            cell.outputs = outputs
            if exec_count is not None:
                cell.execution_count = exec_count

    finally:
        kc.stop_channels()
        km.shutdown_kernel(now=True)

    return nb

# ── Single-notebook runner ────────────────────────────────────────────────────

def run_notebook(
    nb_path: Path,
    output_path: Path,
    dataset: str,
    encoder_dataset,
    overrides: dict | None = None,
    timeout: int = 7200,
) -> bool:
    """Load, inject params, execute, save. Returns True on success."""
    print(f"\n  [{_ts()}] ── {nb_path.name} ──", flush=True)
    t0 = time.time()

    with open(nb_path) as f:
        nb = nbformat.read(f, as_version=4)

    nb = inject_params(nb, dataset, encoder_dataset, overrides=overrides)

    try:
        nb = execute_notebook(nb, nb_path.parent, timeout=timeout)
    except (RuntimeError, TimeoutError) as e:
        elapsed = time.time() - t0
        print(f"  [{_ts()}] ── {nb_path.name}  FAILED ({elapsed/60:.1f} min): {e}",
              flush=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path.with_suffix('.FAILED.ipynb'), 'w') as f:
            nbformat.write(nb, f)
        return False
    except Exception:
        elapsed = time.time() - t0
        print(f"  [{_ts()}] ── {nb_path.name}  FAILED ({elapsed/60:.1f} min)", flush=True)
        traceback.print_exc()
        return False

    elapsed = time.time() - t0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        nbformat.write(nb, f)
    print(f"  [{_ts()}] ── {nb_path.name}  DONE  ({elapsed/60:.1f} min) ──", flush=True)
    return True

# ── Pipeline runner ───────────────────────────────────────────────────────────

def run_pipeline(run_name: str, dry_run: bool = False, timeout: int = 7200,
                 from_nb: str | None = None, until_nb: str | None = None,
                 overrides: dict | None = None) -> bool:
    config          = RUNS[run_name]
    dataset         = config['dataset']
    encoder_dataset = config['encoder_dataset']
    notebooks       = config['notebooks']
    out_dir         = EXECUTED_DIR / run_name

    # Ensure artifacts directory exists
    (REPO_ROOT / 'artifacts' / dataset).mkdir(parents=True, exist_ok=True)

    label = f"dataset={dataset}"
    if encoder_dataset:
        label += f"  encoder={encoder_dataset}"
    if overrides is not None:
        label += "  [quick]"
    if from_nb:
        label += f"  from={from_nb}"
    if until_nb:
        label += f"  until={until_nb}"

    print(f"\n{'='*60}")
    print(f"RUN: {run_name}  ({label})")
    print(f"Output → executed/{run_name}/")
    print(f"{'='*60}")

    skipping = bool(from_nb)
    all_ok = True
    for nb_name in notebooks:
        if skipping:
            if from_nb in nb_name:
                skipping = False
            else:
                print(f"  SKIP  {nb_name}  (before --from)")
                continue
        nb_path = NOTEBOOKS_DIR / nb_name
        if not nb_path.exists():
            print(f"  SKIP  {nb_name}  (not found)")
            continue

        out_path = out_dir / nb_name

        if dry_run:
            enc_label = f" [encoder={encoder_dataset}]" if encoder_dataset else ""
            q_label   = " [quick]" if overrides else ""
            print(f"  DRY   {nb_name}{enc_label}{q_label}")
            if until_nb and until_nb in nb_name:
                break
            continue

        ok = run_notebook(nb_path, out_path, dataset, encoder_dataset,
                          overrides=overrides, timeout=timeout)
        if not ok:
            all_ok = False
            print(f"\n  Pipeline halted at {nb_name}. Fix the error and re-run.")
            break

        if until_nb and until_nb in nb_name:
            print(f"  STOP  (reached --until {until_nb})")
            break

    status = "✓ OK" if all_ok else "✗ FAILED"
    print(f"\n{run_name}: {status}\n")
    return all_ok


# ── Analysis notebooks (always run after pipeline) ────────────────────────────

ANALYSIS_NOTEBOOKS = [
    'analysis/compare_models.ipynb',
    'analysis/compare_mia.ipynb',
    'analysis/compare_evasion.ipynb',
    'analysis/compare_combined.ipynb',
]

def run_analysis(dry_run: bool = False, timeout: int = 600) -> bool:
    """Execute the three cross-dataset analysis notebooks and save to executed/analysis/."""
    out_dir = EXECUTED_DIR / 'analysis'
    print(f"\n{'='*60}")
    print(f"ANALYSIS  (compare_models / compare_mia / compare_evasion)")
    print(f"Output → executed/analysis/")
    print(f"{'='*60}")
    all_ok = True
    for nb_name in ANALYSIS_NOTEBOOKS:
        nb_path = NOTEBOOKS_DIR / nb_name
        if not nb_path.exists():
            print(f"  SKIP  {nb_name}  (not found)")
            continue
        out_path = out_dir / Path(nb_name).name
        if dry_run:
            print(f"  DRY   {nb_name}")
            continue
        ok = run_notebook(nb_path, out_path, dataset='', encoder_dataset=None,
                          overrides=None, timeout=timeout)
        if not ok:
            all_ok = False
    status = "✓ OK" if all_ok else "✗ FAILED"
    print(f"\nAnalysis: {status}\n")
    return all_ok


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('runs', nargs='*',
                        help=f'Run names. Available: {list(RUNS)}')
    parser.add_argument('--quick', action='store_true',
                        help='Smoke-test mode: use reduced epochs/K from the quick: '
                             'block in config.yaml. Results are not thesis-quality; '
                             'use this to verify the pipeline runs on a new machine.')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print what would run without executing')
    parser.add_argument('--list', action='store_true',
                        help='List configured runs and exit')
    parser.add_argument('--timeout', type=int, default=7200,
                        help='Seconds to wait for a single kernel message before '
                             'declaring a hang (default: 7200). Increase for very '
                             'slow cells; 0 = wait forever (risky on OOM).')
    parser.add_argument('--from', dest='from_nb', default=None, metavar='NOTEBOOK',
                        help='Skip all notebooks before this one (e.g. 06b_attack.ipynb). '
                             'Applies to ALL runs specified.')
    parser.add_argument('--until', dest='until_nb', default=None, metavar='NOTEBOOK',
                        help='Stop after this notebook (e.g. 03_train_authenticator.ipynb). '
                             'Applies to ALL runs specified.')
    args = parser.parse_args()

    if args.list:
        print("Configured runs:")
        for name, cfg in RUNS.items():
            enc = cfg['encoder_dataset'] or 'within-dataset'
            nbs = len(cfg['notebooks'])
            print(f"  {name:<20} dataset={cfg['dataset']:<10} encoder={enc:<20} ({nbs} notebooks)")
        sys.exit(0)

    targets = args.runs or list(RUNS.keys())
    unknown = [r for r in targets if r not in RUNS]
    if unknown:
        print(f"Unknown run(s): {unknown}\nAvailable: {list(RUNS)}")
        sys.exit(1)

    cell_timeout = args.timeout or None   # 0 → None → wait forever
    overrides    = load_run_config(quick=args.quick)

    if args.quick:
        print("Quick mode: using reduced parameters from config.yaml quick: block")
        for k, v in overrides.items():
            print(f"  {k} = {v}")

    t_total = time.time()
    results = {}
    for run_name in targets:
        results[run_name] = run_pipeline(
            run_name, dry_run=args.dry_run, timeout=cell_timeout,
            from_nb=args.from_nb, until_nb=args.until_nb,
            overrides=overrides if args.quick else None,
        )

    if not args.until_nb:
        run_analysis(dry_run=args.dry_run, timeout=cell_timeout or 600)

    elapsed_total = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"SUMMARY  ({elapsed_total/60:.1f} min total)")
    print(f"{'='*60}")
    for name, ok in results.items():
        print(f"  {'✓' if ok else '✗'}  {name}")

    if not all(results.values()):
        sys.exit(1)
