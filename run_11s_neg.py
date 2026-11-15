"""C + yolo11s：重置起点 + 15 张厨房负例 + 更大底座，4 个种子。"""
import os
import subprocess
from pathlib import Path

BASE = Path(__file__).resolve().parent
PY = BASE.parent / 'work' / 'board-env' / 'Scripts' / 'python.exe'

for seed in ('73', '42', '7', '11'):
    env = dict(os.environ, SWAP_BACKBONE='yolo11s.pt', SWAP_SEED=seed,
               EXTRA_NEG='experiments/kitchen_neg2_20261115/images', EXTRA_POS='',
               VARIANT_TAG='kitchenneg', PYTHONIOENCODING='utf-8')
    with (BASE / f'c11s_seed{seed}.log').open('w', encoding='utf-8', errors='replace') as handle:
        code = subprocess.call([str(PY), str(BASE / 'train_eval_variant.py')],
                               cwd=str(BASE), env=env, stdout=handle, stderr=subprocess.STDOUT)
    print('11s+neg seed', seed, 'exit', code, flush=True)
