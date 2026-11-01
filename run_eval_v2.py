"""跑 new_source_video_v2 的对照（launch_detached 只能传脚本名，这里把参数包一层）。"""
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.argv = ['evaluate_new_source_v1.py', '--dir', str(BASE / 'experiments' / 'new_source_video_v2'),
            '--out', 'comparison_v1']

import evaluate_new_source_v1 as evaluation  # noqa: E402

if __name__ == '__main__':
    evaluation.main()
