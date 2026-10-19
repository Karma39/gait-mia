"""Execute 06b_attack.ipynb for one or more datasets, injecting DATASET via cell patch."""
import sys
import json
import subprocess
from pathlib import Path

GAIT_DIR = Path(__file__).parent.parent.resolve()
DATASETS = ['whuGAIT', 'ucihar', 'wisdm']


def patch_and_run(dataset: str) -> int:
    nb_src = GAIT_DIR / 'notebooks' / '06b_attack.ipynb'
    nb_tmp = GAIT_DIR / 'notebooks' / f'.06b_{dataset}_tmp.ipynb'
    nb_out = GAIT_DIR / 'executed' / dataset / '06b_attack.ipynb'
    nb_out.parent.mkdir(parents=True, exist_ok=True)

    with open(nb_src) as f:
        nb = json.load(f)

    patched = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'code' and not patched:
            src = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])
            if "DATASET = 'whuGAIT'" in src:
                cell['source'] = (
                    src
                    .replace("sys.path.insert(0, '..')", f"sys.path.insert(0, r'{GAIT_DIR}')")
                    .replace("DATASET = 'whuGAIT'", f"DATASET = '{dataset}'")
                )
                patched = True

    with open(nb_tmp, 'w') as f:
        json.dump(nb, f)

    jupyter_bin = Path(sys.executable).parent / 'jupyter'
    cmd = [
        str(jupyter_bin), 'nbconvert', '--to', 'notebook', '--execute',
        '--ExecutePreprocessor.kernel_name=python3',
        '--ExecutePreprocessor.timeout=36000',   # 10 h
        f'--output={nb_out}', str(nb_tmp),
    ]

    print(f'=== Running NB06b [{dataset}] ===', flush=True)
    result = subprocess.run(cmd, capture_output=False)
    nb_tmp.unlink(missing_ok=True)

    if result.returncode == 0:
        print(f'{dataset} DONE', flush=True)
    else:
        print(f'{dataset} FAILED (exit {result.returncode})', flush=True)
    return result.returncode


if __name__ == '__main__':
    targets = sys.argv[1:] if len(sys.argv) > 1 else DATASETS
    failed = []
    for ds in targets:
        rc = patch_and_run(ds)
        if rc != 0:
            failed.append(ds)
    if failed:
        print(f'FAILED: {failed}')
        sys.exit(1)
    else:
        print('ALL DONE')
