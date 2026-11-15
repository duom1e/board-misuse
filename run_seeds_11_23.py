"""顺序跑 yolo11s 的多个种子（每个：训练 15 轮 + 四组评估）。"""
import os
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
PY = BASE.parent / 'work' / 'board-env' / 'Scripts' / 'python.exe'

for seed in ('11', '23'):
    env = dict(os.environ, SWAP_SEED=seed, PYTHONIOENCODING='utf-8')
    with (BASE / f'seed{seed}.log').open('w', encoding='utf-8', errors='replace') as handle:
        code = subprocess.call([str(PY), str(BASE / 'train_eval_base_swap_seed.py')],
                               cwd=str(BASE), env=env, stdout=handle, stderr=subprocess.STDOUT)
    print('seed', seed, 'exit', code, flush=True)
