"""跑 v6（新场景负例：小学食堂监管检查 / 炒土豆丝）的合体批量，检验误报。"""
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.argv = ['run_combined_batch.py', '--dirs', 'new_source_video_v6', '--out', 'combined_batch_v6']

import run_combined_batch  # noqa: E402

if __name__ == '__main__':
    run_combined_batch.main()
