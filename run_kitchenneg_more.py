"""补 kitchenneg 变体的种子 7、11，确认"重置起点 + 厨房负例"是否稳定。"""
import os
import subprocess
from pathlib import Path

BASE = Path(__file__).resolve().parent
PY = BASE.parent / 'work' / 'board-env' / 'Scripts' / 'python.exe'

for seed in ('7', '11'):
    env = dict(os.environ, SWAP_BACKBONE='yolo11n.pt', SWAP_SEED=seed,
               EXTRA_NEG='experiments/kitchen_neg2_20261115/images', EXTRA_POS='',
               VARIANT_TAG='kitchenneg', PYTHONIOENCODING='utf-8')
    with (BASE / f'rerun_kitchenneg_seed{seed}.log').open('w', encoding='utf-8', errors='replace') as handle:
        code = subprocess.call([str(PY), str(BASE / 'train_eval_variant.py')],
                               cwd=str(BASE), env=env, stdout=handle, stderr=subprocess.STDOUT)
    print('kitchenneg seed', seed, 'exit', code, flush=True)
