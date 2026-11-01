"""跑 v4（切菜合集多场景 / KFC 油炸线）的合体批量对照。"""
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.argv = ['run_combined_batch.py', '--dirs', 'new_source_video_v4', '--out', 'combined_batch_v4']

import run_combined_batch  # noqa: E402

if __name__ == '__main__':
    run_combined_batch.main()
