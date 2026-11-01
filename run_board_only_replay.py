"""用 review_board_only.py 的同一路径跑完 49 帧冻结图，核对单问口径的实际表现。

这里直接复用 run_split_prompts 里保存的 boards 段原始输出做"回放"，
不重复推理；如需重新推理可加 --live（会慢很多）。
判据见 experiments/board_only_v1/decision_rule.json（写在本脚本运行前的手工步骤里）。
"""
import json
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUT = BASE / 'experiments' / 'board_only_v1'
SPLIT = BASE / 'experiments' / 'split_prompts_v1' / 'results.json'


def main():
    rows = json.loads(SPLIT.read_text(encoding='utf-8'))['rows']
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'decision_rule.json').write_text(json.dumps({
        'question': '把案板判断独立成单问后，在 49 帧上表现如何？',
        'method': '回放 split_prompts_v1 中 boards 段的原始输出，走 review_board_only.py 的路径',
        'primary': 'boards 段可解析率（应接近 1.0）',
        'secondary': ['有板命中', '无板误报', '未知帧数'],
        'limits': ['有板样本仅 2 帧', '不算准确率', '同一批帧已多次消费'],
    }, ensure_ascii=False, indent=2), encoding='utf-8')

    summary = {'frames': 0, 'parsed': 0, 'unknown': 0, 'hit': 0, 'miss': 0,
               'false_alarm': 0, 'correct_silent': 0, 'excluded': 0}
    details = []
    for index, row in enumerate(rows):
        name = row['frame']
        raw = row['raw_boards']
        raw_file = OUT / f'raw_{index:02d}.txt'
        raw_file.write_text(raw, encoding='utf-8')
        dest = OUT / 'runs' / f'{index:02d}_{name[:-4]}'
        if dest.exists():
            continue
        command = [sys.executable, str(BASE / 'review_board_only.py'),
                   str(Path(row['label'].get('image', '')) if row['label'].get('image') else ''),
                   '--output-dir', str(dest), '--response-file', str(raw_file)]
        # 图片路径来自原始实验记录
        image_path = BASE / 'experiments' / row['source'] / 'frames' / name
        command[2] = str(image_path)
        completed = subprocess.run(command, cwd=BASE, capture_output=True, text=True)
        if completed.returncode != 0:
            details.append({'frame': name, 'error': completed.stderr[-300:]})
            continue
        result = json.loads((dest / 'result.json').read_text(encoding='utf-8'))
        state = result['board_state']
        label = row['label']
        summary['frames'] += 1
        summary['parsed'] += 1 if result['output_validation']['valid'] else 0
        summary['unknown'] += 1 if state == 'unknown' else 0
        if label.get('usable') is False:
            summary['excluded'] += 1
        elif label.get('board_present') == 'yes':
            summary['hit'] += state == 'candidate_present'
            summary['miss'] += state != 'candidate_present'
        elif label.get('board_present') == 'no':
            summary['false_alarm'] += state == 'candidate_present'
            summary['correct_silent'] += state != 'candidate_present'
        details.append({'frame': name, 'state': state, 'valid': result['output_validation']['valid'],
                        'label': label.get('board_present'), 'usable': label.get('usable')})
    (OUT / 'summary.json').write_text(json.dumps({'summary': summary, 'details': details},
                                                 ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
