"""补 yolo11n 从 COCO 预训练起步的其余种子（73 已完成），作为 yolo11s 的公平对照。"""
import os
import subprocess
from pathlib import Path

BASE = Path(__file__).resolve().parent
PY = BASE.parent / 'work' / 'board-env' / 'Scripts' / 'python.exe'

for seed in ('42', '7', '11', '23'):
    env = dict(os.environ, SWAP_SEED=seed, SWAP_BACKBONE='yolo11n.pt', PYTHONIOENCODING='utf-8')
    with (BASE / f'n11_seed{seed}.log').open('w', encoding='utf-8', errors='replace') as handle:
        code = subprocess.call([str(PY), str(BASE / 'train_eval_swap_param.py')],
                               cwd=str(BASE), env=env, stdout=handle, stderr=subprocess.STDOUT)
    print('yolo11n seed', seed, 'exit', code, flush=True)
