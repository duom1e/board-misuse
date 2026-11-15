"""补 yolo11n 从 COCO 预训练起步的 5 个种子，作为 yolo11s 的公平对照。

注意：原基线 pool4_positive15_seed73 是从**已训练过的** board_v2_real 权重起步，
而 yolo11s 是从 COCO 预训练起步，两者起点不同。本脚本把 yolo11n 也改成 COCO 起步，
在同一起点上比较容量。
"""
import os
import subprocess
from pathlib import Path

BASE = Path(__file__).resolve().parent
PY = BASE.parent / 'work' / 'board-env' / 'Scripts' / 'python.exe'

for seed in ('73', '42', '7', '11', '23'):
    env = dict(os.environ, SWAP_SEED=seed, SWAP_BACKBONE='yolo11n.pt', PYTHONIOENCODING='utf-8')
    with (BASE / f'n11_seed{seed}.log').open('w', encoding='utf-8', errors='replace') as handle:
        code = subprocess.call([str(PY), str(BASE / 'train_eval_swap_param.py')],
                               cwd=str(BASE), env=env, stdout=handle, stderr=subprocess.STDOUT)
    print('yolo11n seed', seed, 'exit', code, flush=True)
