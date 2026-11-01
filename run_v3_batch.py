"""跑 v3（砧板岗位 / 冷盘切配）的合体批量对照。"""
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.argv = ['run_combined_batch.py', '--dirs', 'new_source_video_v3', '--out', 'combined_batch_v3']

import run_combined_batch  # noqa: E402

if __name__ == '__main__':
    run_combined_batch.main()
