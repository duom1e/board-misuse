"""在新基线（COCO 起点）上重跑三组样本增删实验，各 2 个种子。

对照含义：
  plain        = 只有 270 条基线（B 组，已有 5 种子）
  kitchenneg   = +15 张厨房/锅具负例
  woodpos      = +3 张木砧板正例（各 2 次）
"""
import os
import subprocess
from pathlib import Path

BASE = Path(__file__).resolve().parent
PY = BASE.parent / 'work' / 'board-env' / 'Scripts' / 'python.exe'

VARIANTS = [
    ('kitchenneg', 'experiments/kitchen_neg2_20261115/images', ''),
    ('woodpos', '', 'experiments/wood_positive_20261115'),
]
SEEDS = ('73', '42')

for tag, neg, pos in VARIANTS:
    for seed in SEEDS:
        env = dict(os.environ, SWAP_BACKBONE='yolo11n.pt', SWAP_SEED=seed,
                   EXTRA_NEG=neg, EXTRA_POS=pos, VARIANT_TAG=tag, PYTHONIOENCODING='utf-8')
        log = BASE / f'rerun_{tag}_seed{seed}.log'
        with log.open('w', encoding='utf-8', errors='replace') as handle:
            code = subprocess.call([str(PY), str(BASE / 'train_eval_variant.py')],
                                   cwd=str(BASE), env=env, stdout=handle, stderr=subprocess.STDOUT)
        print(tag, 'seed', seed, 'exit', code, flush=True)
