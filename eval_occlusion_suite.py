"""阶段 0 评估套件 v1：在遮挡盲测集上跑"单帧 YOLO"基线（A 组）。

输出（一次运行全给）：
  · 按遮挡档分的召回：R_none / R_mild（severe 目前 0 帧，脚本会显式标注不可算）
  · 无板帧误报率
  · 命中中的"位置可用"比例（框面积 <0.6 视为可用）
  · 按来源分组的结果（避免同源相关帧被当成独立样本）
  · 逐帧 json，便于复核

不做训练、不改阈值，只用现有候选模型。
用法：python eval_occlusion_suite.py [weights] [imgsz]
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

from ultralytics import YOLO

BASE = Path(__file__).resolve().parent
OUT = BASE / 'experiments' / 'occlusion_phase0'
DEFAULT_WEIGHTS = BASE / 'training_runs' / 'board_neg15_v1' / 'weights' / 'last.pt'
IOU = 0.5


def main():
    weights = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_WEIGHTS
    size = int(sys.argv[2]) if len(sys.argv) > 2 else 640
    labels = json.loads((OUT / 'labels_blind.json').read_text(encoding='utf-8'))['labels']
    model = YOLO(str(weights))

    rows = []
    for name, label in sorted(labels.items()):
        path = OUT / 'frames' / name
        if not path.exists():
            continue
        result = model.predict(str(path), imgsz=size, rect=True, conf=.05, device=0, verbose=False)[0]
        boxes = result.boxes
        height, width = result.orig_shape
        entries = [{'score': round(float(s), 4),
                    'area': round(((b[2] - b[0]) * (b[3] - b[1])) / float(width * height), 3),
                    'box': [round(float(v), 1) for v in b]}
                   for b, s in zip(boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist())]
        rows.append({'frame': name, 'source': label.get('source_bvid'),
                     'label': label['board_present'], 'level': label['occlusion_level'],
                     'seconds': label.get('seconds'), 'boxes': entries})

    summary = {}
    for threshold in (0.25, 0.5):
        stats = defaultdict(lambda: {'n': 0, 'hit': 0, 'usable': 0})
        for row in rows:
            key = row['level'] if row['label'] == 'yes' else ('no_board' if row['label'] == 'no' else 'unclear')
            best = max([b['score'] for b in row['boxes']], default=0.0)
            usable = any(b['score'] >= threshold and b['area'] < 0.6 for b in row['boxes'])
            entry = stats[key]
            entry['n'] += 1
            if key == 'no_board':
                if best >= threshold:
                    entry['hit'] += 1  # 误报计数
            elif key == 'unclear':
                pass
            else:
                if best >= threshold:
                    entry['hit'] += 1
                    if usable:
                        entry['usable'] += 1
        summary[f'conf_{threshold}'] = {k: dict(v) for k, v in stats.items()}

    per_source = defaultdict(lambda: {'n': 0, 'hit': 0})
    for row in rows:
        if row['label'] != 'yes':
            continue
        best = max([b['score'] for b in row['boxes']], default=0.0)
        per_source[row['source']]['n'] += 1
        if best >= 0.25:
            per_source[row['source']]['hit'] += 1

    report = {'weights': str(weights), 'imgsz': size, 'frames': len(rows),
              'note': 'severe 档当前为 0 帧，R_severe 不可计算；unclear 帧不计分',
              'summary': summary, 'per_source_conf_0.25': dict(per_source)}
    (OUT / 'baseline_A_eval.json').write_text(json.dumps({'report': report, 'rows': rows},
                                                         ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
