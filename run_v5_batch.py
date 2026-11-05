"""跑 v5（中景/广角后厨）的合体批量对照，重点看中景帧的位置可用性。"""
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.argv = ['run_combined_batch.py', '--dirs', 'new_source_video_v5', '--out', 'combined_batch_v5']

import run_combined_batch  # noqa: E402

if __name__ == '__main__':
    run_combined_batch.main()
