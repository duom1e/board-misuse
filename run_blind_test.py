"""盲测评估：用候选模型 board_neg15_v1 跑 blind_test_v1 的 20 帧，统计命中/误报/漏检。

不训练、不调参、不看结果改标签。conf .25 与 .5 两档。
"""
import json
from pathlib import Path

from ultralytics import YOLO

BASE = Path(__file__).resolve().parent
OUT = BASE / 'experiments' / 'blind_test_v1'
WEIGHTS = BASE / 'training_runs' / 'board_neg15_v1' / 'weights' / 'last.pt'
IOU = 0.5


def area_fraction(box, width, height):
    return ((box[2] - box[0]) * (box[3] - box[1])) / float(width * height)


def main():
    labels = json.loads((OUT / 'labels_blind.json').read_text(encoding='utf-8'))['labels']
    model = YOLO(str(WEIGHTS))
    rows = []
    for name, label in labels.items():
        path = OUT / 'frames' / name
        result = model.predict(str(path), imgsz=640, rect=True, conf=.05, device=0, verbose=False)[0]
        boxes = result.boxes
        height, width = result.orig_shape
        pairs = [{'box': [round(v, 1) for v in b], 'score': round(s, 4),
                  'area': round(area_fraction(b, width, height), 3)}
                 for b, s in zip(boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist())]
        rows.append({'frame': name, 'label': label['board_present'], 'note': label['note'], 'boxes': pairs})
        print(name, label['board_present'], [(p['score'], p['area']) for p in pairs], flush=True)

    summary = []
    for threshold in (0.25, 0.5):
        present = [r for r in rows if r['label'] == 'yes']
        absent = [r for r in rows if r['label'] == 'no']
        hits = sum(1 for r in present if any(p['score'] >= threshold for p in r['boxes']))
        false_alarms = sum(1 for r in absent if any(p['score'] >= threshold for p in r['boxes']))
        usable = sum(1 for r in present
                     if any(p['score'] >= threshold and p['area'] < 0.6 for p in r['boxes']))
        summary.append({'threshold': threshold, 'board_frames': len(present), 'hits': hits,
                        'position_usable': usable, 'no_board_frames': len(absent),
                        'false_alarms': false_alarms})
    result = {'weights': str(WEIGHTS), 'imgsz': 640, 'summary': summary}
    (OUT / 'results.json').write_text(json.dumps({'summary': summary, 'rows': rows}, ensure_ascii=False, indent=2),
                                      encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
