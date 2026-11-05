"""合体入口的批量版：49 帧冻结图一次跑完，统计"两线一致 / 冲突 / 位置不可用"分布。

模型只加载一次；每帧输出 result.json 精简要件，汇总写到 experiments/combined_batch_v1/summary.json。
判据先写死：这是分布统计，不是准确率。
"""
import argparse
import json
from pathlib import Path

import review_combined as rc

BASE = Path(__file__).resolve().parent
OUT = BASE / 'experiments' / 'combined_batch_v1'
DIRS = [BASE / 'experiments' / 'new_source_video_v1', BASE / 'experiments' / 'new_source_video_v2']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dirs', default='', help='comma separated experiment dir names')
    parser.add_argument('--out', default='combined_batch_v1')
    args = parser.parse_args()
    global OUT, DIRS
    if args.dirs:
        DIRS = [BASE / 'experiments' / name for name in args.dirs.split(',') if name]
    OUT = BASE / 'experiments' / args.out
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "results").mkdir()
    (OUT / 'decision_rule.json').write_text(json.dumps({
        'question': '合体入口在这 49 帧上的行为分布：两线一致、冲突、位置不可用各占多少？',
        'not_a_metric': '这不是准确率；有板样本仅 2 帧',
        'primary': 'board_presence 分布 + position_usable 比例 + 两线冲突比例',
        'frozen_before_run': True,
    }, ensure_ascii=False, indent=2), encoding='utf-8')

    detectors = rc.load_detectors()
    vlm = rc.load_vlm()
    rows = []
    for directory in DIRS:
        manifest = json.loads((directory / 'frames_manifest.json').read_text(encoding='utf-8'))
        labels = json.loads((directory / 'labels_blind.json').read_text(encoding='utf-8'))['labels']
        for record in manifest['frames']:
            name = Path(record['file']).name
            result, _ = rc.analyze(record['file'], detectors, vlm)
            (OUT / 'results' / (directory.name + '_' + name + '.json')).write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
            label = labels.get(name, {})
            rows.append({
                'frame': name, 'source': directory.name,
                'board_presence': result['board_presence'],
                'board_source': result['board_source'],
                'position_usable': result['position_usable'],
                'vlm_state': result['lines']['vlm']['board_state'],
                'vlm_valid': result['lines']['vlm']['valid'],
                'yolo_boards': len(result['lines']['detector']['board_boxes']),
                'food_boxes': len(result['food_from_detector']),
                'food_locations': [r['location'] for r in result['food_regions']],
                'label_board': label.get('board_present'),
                'label_usable': label.get('usable'),
            })
            print(name, result['board_presence'], result['board_source'], result['position_usable'], flush=True)

    counts = {}
    for row in rows:
        counts[row['board_presence']] = counts.get(row['board_presence'], 0) + 1
    summary = {
        'frames': len(rows),
        'board_presence_counts': counts,
        'position_usable': sum(1 for r in rows if r['position_usable']),
        'vlm_parse_failures': sum(1 for r in rows if not r['vlm_valid']),
        'conflicts': sum(1 for r in rows if r['board_presence'] in
                         ('uncertain_vlm_no_yolo_yes', 'present_vlm_box_coarse_using_yolo',
                          'present_position_coarse', 'unknown')),
        'with_board_label': sum(1 for r in rows if r['label_board'] == 'yes'),
        'note': '分布统计，不是准确率',
    }
    (OUT / 'summary.json').write_text(json.dumps({'summary': summary, 'rows': rows},
                                                 ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
