"""Cheese-specific comparison of the v1 and v3 models on identical held-out images.

Why: v3 passed four of five pre-registered criteria and missed only "cheese still detected
at conf >= 0.5" (it reported animal_food 0.335 on one image where v1 said 0.750). One image
is not evidence either way, so this measures how often each model finds Cheese boxes across
all available held-out cheese images.

Pre-registered interpretation (written before running):
  - if v3 matched-cheese recall is not clearly below v1 (allowed drop <= 0.03), the single
    image was sampling noise and v3 may replace v1 in the registry
  - otherwise v1 stays registered
"""
import csv
import json
from pathlib import Path

import numpy as np
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent
RAW = BASE.parent / 'work' / 'datasets'
V3 = BASE / 'food_taxonomy_v3'
CHEESE = '/m/01nkt'
CAP = 120
CONF = 0.5
TOLERANCE = 0.03


def iou(a, b):
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def cheese_images():
    test_ids = {Path(p).stem for p in (V3 / 'test.txt').read_text().split() if p.strip()}
    annotations = {}
    with (RAW / 'oi_test.csv').open(encoding='utf-8', newline='') as f:
        reader = csv.reader(f)
        idx = {name: i for i, name in enumerate(next(reader))}
        for row in reader:
            image_id = row[idx['ImageID']]
            if image_id not in test_ids or row[idx['LabelName']] != CHEESE:
                continue
            if row[idx['IsGroupOf']] != '0' or row[idx['IsDepiction']] != '0':
                continue
            annotations.setdefault(image_id, []).append(
                tuple(float(row[idx[k]]) for k in ('XMin', 'YMin', 'XMax', 'YMax')))
    return sorted(annotations)[:CAP], annotations


DEFAULT_RUNS = ['v1=food_taxonomy_v1', 'v3=food_taxonomy_v3']


def resolve(spec):
    name, _, run = spec.partition('=')
    run = run or name
    path = BASE / 'training_runs' / run / 'weights' / 'best.pt'
    return name, path


def main(conf=CONF, output='cheese_recall_comparison.json', runs=None):
    images, annotations = cheese_images()
    from PIL import Image
    models = {}
    for spec in (runs or DEFAULT_RUNS):
        model_name, path = resolve(spec)
        models[model_name] = YOLO(str(path))
    totals = {key: {'cheese_boxes': 0, 'matched': 0, 'images': 0, 'images_with_hit': 0} for key in models}
    per_image = []
    for image_id in images:
        path = V3 / 'images' / 'test' / f'{image_id}.jpg'
        if not path.exists():
            continue
        with Image.open(path) as im:
            width, height = im.size
        cheese_boxes = [(x1 * width, y1 * height, x2 * width, y2 * height)
                        for x1, y1, x2, y2 in annotations[image_id]]
        row = {'image': image_id, 'cheese_boxes': len(cheese_boxes)}
        for model_name, model in models.items():
            result = model.predict(source=str(path), conf=conf, imgsz=640, device=0, verbose=False)[0]
            preds = result.boxes.xyxy.cpu().numpy() if result.boxes is not None and len(result.boxes) else np.zeros((0, 4))
            classes = result.boxes.cls.cpu().numpy().astype(int) if len(preds) else np.zeros(0, dtype=int)
            pred_boxes = [tuple(p) for p, c in zip(preds, classes) if model.names[c] == 'animal_food']
            matched = sum(1 for box in cheese_boxes if any(iou(p, box) >= 0.5 for p in pred_boxes))
            totals[model_name]['cheese_boxes'] += len(cheese_boxes)
            totals[model_name]['matched'] += matched
            totals[model_name]['images'] += 1
            totals[model_name]['images_with_hit'] += 1 if matched else 0
            row[model_name] = {'detections': len(pred_boxes), 'matched_cheese': matched}
        per_image.append(row)
    for key in totals:
        entry = totals[key]
        entry['recall_proxy'] = round(entry['matched'] / entry['cheese_boxes'], 4) if entry['cheese_boxes'] else None
    baseline = totals.get('v1', {}).get('recall_proxy') or 0
    deltas = {key: round((value['recall_proxy'] or 0) - baseline, 4) for key, value in totals.items()}
    payload = {
        'purpose': 'Compare animal_food detection on images that contain Cheese ground truth, identical for every model.',
        'images': len(per_image), 'confidence': conf, 'iou': 0.5,
        'note': 'animal_food predictions are matched against Cheese boxes only; other animal boxes in the same image are ignored.',
        'totals': totals, 'recall_delta_vs_v1': deltas,
        'pre_registered_tolerance': TOLERANCE,
        'per_image': per_image,
    }
    (BASE / output).write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(json.dumps({k: payload[k] for k in ('images', 'totals', 'recall_delta_vs_v1')},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=float, default=CONF)
    parser.add_argument('--output', default='cheese_recall_comparison.json')
    parser.add_argument('--post-hoc', action='store_true', help='mark a run made after the primary result')
    parser.add_argument('--runs', nargs='+', default=DEFAULT_RUNS,
                        help='name=run_dir (default: v1=food_taxonomy_v1 v3=food_taxonomy_v3)')
    args = parser.parse_args()
    CONF = args.conf
    main(args.conf, args.output, args.runs)
    if args.post_hoc:
        path = BASE / args.output
        payload = json.loads(path.read_text())
        payload['post_hoc'] = True
        payload['label'] = 'Run after the primary decision was recorded; used only to separate calibration from capability.'
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

