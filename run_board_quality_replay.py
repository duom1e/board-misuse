"""用质量感知版 review_board_only.py 回放 49 帧，统计"位置可用"比例。"""
import json
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUT = BASE / 'experiments' / 'board_only_v2_quality'
SPLIT = BASE / 'experiments' / 'split_prompts_v1' / 'results.json'


def main():
    rows = json.loads(SPLIT.read_text(encoding='utf-8'))['rows']
    (OUT / 'runs').mkdir(parents=True, exist_ok=True)
    summary = {'frames': 0, 'candidate': 0, 'position_usable': 0, 'position_coarse': 0,
               'no_board_detected': 0, 'unknown': 0}
    details = []
    for index, row in enumerate(rows):
        name = row['frame']
        raw_file = BASE / 'experiments' / 'board_only_v1' / f'raw_{index:02d}.txt'
        if not raw_file.exists():
            continue
        image_path = BASE / 'experiments' / row['source'] / 'frames' / name
        dest = OUT / 'runs' / f'{index:02d}_{name[:-4]}'
        if dest.exists():
            continue
        command = [sys.executable, str(BASE / 'review_board_only.py'), str(image_path),
                   '--output-dir', str(dest), '--response-file', str(raw_file)]
        completed = subprocess.run(command, cwd=BASE, capture_output=True, text=True)
        if completed.returncode != 0:
            details.append({'frame': name, 'error': completed.stderr[-200:]})
            continue
        result = json.loads((dest / 'result.json').read_text(encoding='utf-8'))
        state = result['board_state']
        summary['frames'] += 1
        if state == 'candidate_present':
            summary['candidate'] += 1
            if result['position_usable']:
                summary['position_usable'] += 1
            else:
                summary['position_coarse'] += 1
        elif state == 'no_board_detected':
            summary['no_board_detected'] += 1
        else:
            summary['unknown'] += 1
        details.append({'frame': name, 'state': state,
                        'position_usable': result['position_usable'],
                        'next_action': result['next_action'],
                        'label': row['label'].get('board_present')})
    (OUT / 'summary.json').write_text(json.dumps({'summary': summary, 'details': details},
                                                 ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
