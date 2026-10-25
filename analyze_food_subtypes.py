"""Per-subtype breakdown of what the food models actually detect.

Motivation: every 3-class run scores badly on `animal_food` (AP50 0.14) while `other_food`
reaches 0.53-0.57. Either the models are weak, or `animal_food` is not one visual class:
it lumps Seafood, Shellfish, Shrimp, Egg and Cheese together. This measures detection rate
per Open Images source class on the identical v3 test split, for several models at once.

Method: match each GT box in the v3 test split to its Open Images source class (via the
official test CSV), then check whether an `animal_food` / `plant_food` prediction of the
owning bucket overlaps it at IoU >= 0.5 with confidence >= the threshold. Matching is
class-agnostic inside a bucket, so a seafood box can be matched by any animal_food box.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent
RAW = BASE.parent / 'work' / 'datasets'
V3 = BASE / 'food_taxonomy_v3'
BUCKET = {}
for mid in ['/m/0f4s2w', '/m/0hkxq', '/m/0fbw6', '/m/0fj52s', '/m/015x4r', '/m/015x5n', '/m/07j87']:
    BUCKET[mid] = 'plant_food'
for mid in ['/m/06nwz', '/m/0fbdv', '/m/0ll1f78', '/m/033cnk', '/m/01nkt']:
    BUCKET[mid] = 'animal_food'
for mid in ['/m/09728', '/m/07030', '/m/0cdn1', '/m/0jy4k']:
    BUCKET[mid] = 'other_food'
NAMES = {'/m/0f4s2w': 'Vegetable', '/m/0hkxq': 'Broccoli', '/m/0fbw6': 'Cabbage', '/m/0fj52s': 'Carrot',
         '/m/015x4r': 'Cucumber', '/m/015x5n': 'Radish', '/m/07j87': 'Tomato', '/m/06nwz': 'Seafood',
         '/m/0fbdv': 'Shellfish', '/m/0ll1f78': 'Shrimp', '/m/033cnk': 'Egg', '/m/01nkt': 'Cheese',
         '/m/09728': 'Bread', '/m/07030': 'Sushi', '/m/0cdn1': 'Hamburger', '/m/0jy4k': 'Doughnut'}


def iou(a, b):
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def test_boxes():
    ids = {Path(p).stem for p in (V3 / 'test.txt').read_text().split() if p.strip()}
    boxes = defaultdict(list)
    with (RAW / 'oi_test.csv').open(encoding='utf-8', newline='') as f:
        reader = csv.reader(f)
        idx = {name: i for i, name in enumerate(next(reader))}
        for row in reader:
            image_id = row[idx['ImageID']]
            mid = row[idx['LabelName']]
            if image_id not in ids or mid not in BUCKET:
                continue
            if row[idx['IsGroupOf']] != '0' or row[idx['IsDepiction']] != '0':
                continue
            boxes[image_id].append((BUCKET[mid], mid,
                                    tuple(float(row[idx[k]]) for k in ('XMin', 'YMin', 'XMax', 'YMax'))))
    return boxes


def main(runs, conf, output='food_subtype_analysis.json'):
    boxes = test_boxes()
    models = {}
    for spec in runs:
        name, _, run = spec.partition('=')
        models[name] = YOLO(str(BASE / 'training_runs' / (run or name) / 'weights' / 'best.pt'))
    tallies = {name: defaultdict(lambda: {'boxes': 0, 'matched': 0}) for name in models}
    per_image = []
    for image_id in sorted(boxes):
        path = V3 / 'images' / 'test' / f'{image_id}.jpg'
        if not path.exists():
            continue
        with Image.open(path) as im:
            width, height = im.size
        gt = [(bucket, mid, (b[0] * width, b[1] * height, b[2] * width, b[3] * height))
              for bucket, mid, b in boxes[image_id]]
        row = {'image': image_id}
        for name, model in models.items():
            result = model.predict(source=str(path), conf=conf, imgsz=640, device=0, verbose=False)[0]
            preds = result.boxes.xyxy.cpu().numpy() if result.boxes is not None and len(result.boxes) else np.zeros((0, 4))
            classes = result.boxes.cls.cpu().numpy().astype(int) if len(preds) else np.zeros(0, dtype=int)
            by_bucket = defaultdict(list)
            for pred, cls in zip(preds, classes):
                by_bucket[model.names[cls]].append(tuple(pred))
            matched_total = 0
            for bucket, mid, box in gt:
                hit = any(iou(p, box) >= 0.5 for p in by_bucket.get(bucket, []))
                tallies[name][mid]['boxes'] += 1
                tallies[name][mid]['matched'] += 1 if hit else 0
                matched_total += 1 if hit else 0
            row[name] = {'gt': len(gt), 'matched': matched_total}
        per_image.append(row)
    summary = {}
    for name in models:
        summary[name] = {}
        for mid, stats in sorted(tallies[name].items()):
            summary[name][NAMES[mid]] = {
                'bucket': BUCKET[mid], 'boxes': stats['boxes'], 'matched': stats['matched'],
                'detection_rate': round(stats['matched'] / stats['boxes'], 4) if stats['boxes'] else None}
    payload = {
        'split': str(V3 / 'test'),
        'confidence': conf, 'iou': 0.5,
        'note': 'Detection rate = share of ground-truth boxes of that Open Images class matched by any prediction of the same bucket. Not AP.',
        'models': {spec.partition('=')[0]: str(BASE / 'training_runs' / (spec.partition('=')[2] or spec.partition('=')[0]) / 'weights' / 'best.pt') for spec in runs},
        'per_class': summary, 'per_image': per_image,
    }
    (BASE / output).write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    for name in summary:
        print(name, json.dumps({k: v['detection_rate'] for k, v in summary[name].items()}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', nargs='+', default=['v1=food_taxonomy_v1', 'v3=food_taxonomy_v3', 'v4=food_taxonomy_v4s'])
    parser.add_argument('--conf', type=float, default=0.5)
    parser.add_argument('--output', default='food_subtype_analysis.json')
    args = parser.parse_args()
    main(args.runs, args.conf, args.output)

