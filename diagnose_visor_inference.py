"""Diagnostic: does inference resolution (or tiling) explain the zero recall on VISOR frames?

The 13 VISOR frames are 1920x1080 egocentric kitchen shots where the board is often small and
oblique. At imgsz 640 a 1920-wide frame is downscaled 3x, so a board ~250 px wide arrives as
~80 px. This measures the three frozen models at several inference settings, on the frames
already viewed in round 22 (so this is development/diagnostic data, NOT a fresh test).

Pre-declared rule: a setting that looks good here must be confirmed on frames from participants
that have never been used before it can influence any default. This script never touches
active_models.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent
PROBE = BASE / 'visor_board_probe'
CONF = 0.5
IOU = 0.5
CONFIGS = [('640', 640, False), ('960', 960, False), ('1280', 1280, False), ('640_tiles2x2', 640, True)]


def load_gt(label_path, width, height):
    boxes = []
    for line in label_path.read_text().splitlines():
        if not line.strip():
            continue
        _, cx, cy, w, h = (float(v) for v in line.split())
        boxes.append(((cx - w / 2) * width, (cy - h / 2) * height,
                      (cx + w / 2) * width, (cy + h / 2) * height))
    return boxes


def iou(a, b):
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def nms(boxes, scores, threshold=0.5):
    order = np.argsort(-np.asarray(scores))
    keep = []
    for index in order:
        if all(iou(boxes[index], boxes[other]) <= threshold for other in keep):
            keep.append(index)
    return keep


def predict_tiled(model, path, imgsz, tiles=2, overlap=0.2):
    with Image.open(path) as handle:
        image = handle.convert('RGB')
    width, height = image.size
    boxes, scores = [], []
    tile_w = width / (tiles - (tiles - 1) * overlap)
    tile_h = height / (tiles - (tiles - 1) * overlap)
    step_x = tile_w * (1 - overlap)
    step_y = tile_h * (1 - overlap)
    for row in range(tiles):
        for col in range(tiles):
            x1 = min(col * step_x, width - tile_w)
            y1 = min(row * step_y, height - tile_h)
            crop = image.crop((int(x1), int(y1), int(x1 + tile_w), int(y1 + tile_h)))
            result = model.predict(source=np.array(crop), conf=CONF, imgsz=imgsz, device=0, verbose=False)[0]
            if result.boxes is None or not len(result.boxes):
                continue
            for box, score in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist()):
                boxes.append((box[0] + x1, box[1] + y1, box[2] + x1, box[3] + y1))
                scores.append(float(score))
    if not boxes:
        return []
    keep = nms(boxes, scores)
    return [boxes[i] for i in keep]


def predict_plain(model, path, imgsz):
    result = model.predict(source=str(path), conf=CONF, imgsz=imgsz, device=0, verbose=False)[0]
    if result.boxes is None or not len(result.boxes):
        return []
    return [tuple(box) for box in result.boxes.xyxy.cpu().tolist()]


def evaluate(boxes_by_image, gt_by_image):
    tp = fp = fn = 0
    for name, gt in gt_by_image.items():
        used = [False] * len(gt)
        for box in sorted(boxes_by_image.get(name, []), key=lambda b: -0.0):
            best, best_j = 0.0, -1
            for j, g in enumerate(gt):
                if used[j]:
                    continue
                value = iou(box, g)
                if value > best:
                    best, best_j = value, j
            if best_j >= 0 and best >= IOU:
                used[best_j] = True
                tp += 1
            else:
                fp += 1
        fn += used.count(False)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return {'tp': tp, 'fp': fp, 'fn': fn,
            'precision': round(precision, 4) if precision is not None else None,
            'recall': round(recall, 4) if recall is not None else None,
            'precision_defined': precision is not None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=['default_board', 'expanded_s'])
    parser.add_argument('--output', default='visor_inference_diagnostic.json')
    args = parser.parse_args()

    registry = json.loads((BASE / 'active_models.json').read_text())
    weights = {
        'default_board': registry['board']['weights'],
        'expanded_s': str(BASE / 'experiments' / 'board_expanded_s_reviewed' / 'training' / 'weights' / 'best.pt'),
        'board_lvis': registry['board_lvis']['weights'],
    }
    images = sorted((PROBE / 'images').glob('*.jpg'))
    gt = {}
    for path in images:
        with Image.open(path) as image:
            width, height = image.size
        gt[path.stem] = load_gt(PROBE / 'labels' / f'{path.stem}.txt', width, height)
    payload = {
        'purpose': 'Diagnose whether inference resolution or tiling explains zero recall on 13 VISOR egocentric frames.',
        'data_status': 'frames already viewed in round 22; development/diagnostic only, not a fresh test',
        'confidence': CONF, 'iou': IOU, 'configs': [c[0] for c in CONFIGS],
        'images': len(images), 'gt_boxes': sum(len(v) for v in gt.values()),
        'pre_declared_rule': 'any setting that helps here must be confirmed on participants never used before it may affect a default',
        'results': {},
    }
    for name in args.models:
        path = Path(weights[name])
        if not path.exists():
            payload['results'][name] = {'error': f'missing weights: {path}'}
            continue
        model = YOLO(str(path))
        per_config = {}
        for label, imgsz, tiled in CONFIGS:
            detected = {}
            for image_path in images:
                boxes = predict_tiled(model, image_path, imgsz) if tiled else predict_plain(model, image_path, imgsz)
                detected[image_path.stem] = boxes
            per_config[label] = evaluate(detected, gt)
            per_config[label]['images_with_detection'] = sum(1 for v in detected.values() if v)
            payload['results'][name] = per_config
        print(name, json.dumps(per_config, ensure_ascii=False), flush=True)

    best_name = None
    best_recall = -1.0
    for name, per_config in payload['results'].items():
        if 'error' in per_config:
            continue
        for label, metrics in per_config.items():
            if (metrics['recall'] or 0) > best_recall:
                best_recall = metrics['recall'] or 0
                best_name = f'{name}@{label}'
    payload['best_setting'] = {'name': best_name, 'recall': best_recall}
    (BASE / args.output).write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print('best', best_name, best_recall)


if __name__ == '__main__':
    main()
