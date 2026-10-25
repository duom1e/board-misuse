"""Calibrate the confidence threshold instead of assuming 0.5.

Confidence 0.5 was never justified; the subtype analysis showed hit rates triple at 0.25.
This collects every prediction at a low floor on a chosen split, then sweeps thresholds and
reports precision/recall/F1 for each, per class and micro-averaged.

Calibration split should be validation. Test is only for confirming the chosen point, so the
test numbers stay honest.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent
FLOOR = 0.01
THRESHOLDS = [round(0.05 + 0.05 * i, 2) for i in range(18)]  # 0.05 .. 0.90


def iou_matrix(pred, gt):
    if not len(pred) or not len(gt):
        return np.zeros((len(pred), len(gt)))
    px1, py1, px2, py2 = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3], pred[:, 3:4]
    gx1, gy1, gx2, gy2 = gt[None, :, 0], gt[None, :, 1], gt[None, :, 2], gt[None, :, 3]
    iw = np.clip(np.minimum(px2, gx2) - np.maximum(px1, gx1), 0, None)
    ih = np.clip(np.minimum(py2, gy2) - np.maximum(py1, gy1), 0, None)
    inter = iw * ih
    union = ((px2 - px1) * (py2 - py1)) + ((gx2 - gx1) * (gy2 - gy1)) - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def load_gt(label_path, width, height):
    rows = []
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            if not line.strip():
                continue
            cls, x, y, w, h = line.split()
            x, y, w, h = (float(v) for v in (x, y, w, h))
            rows.append((int(cls), ((x - w / 2) * width, (y - h / 2) * height,
                                    (x + w / 2) * width, (y + h / 2) * height)))
    return rows


def collect(model, dataset, split, imgsz=640):
    images = [Path(p) for p in (dataset / f'{split}.txt').read_text().split() if p.strip()]
    records = []
    for path in images:
        result = model.predict(source=str(path), conf=FLOOR, imgsz=imgsz, device=0, verbose=False)[0]
        height, width = result.orig_shape[:2]
        gt = load_gt(dataset / 'labels' / split / f'{path.stem}.txt', width, height)
        if result.boxes is not None and len(result.boxes):
            pred = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
        else:
            pred, scores, classes = np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)
        records.append({'gt': gt, 'pred': pred, 'scores': scores, 'classes': classes})
    return records


def evaluate(records, threshold, class_count):
    tp = np.zeros(class_count, dtype=int)
    fp = np.zeros(class_count, dtype=int)
    fn = np.zeros(class_count, dtype=int)
    for record in records:
        keep = record['scores'] >= threshold
        pred, classes = record['pred'][keep], record['classes'][keep]
        gt = record['gt']
        used = [False] * len(gt)
        order = np.argsort(-record['scores'][keep]) if keep.any() else []
        for index in order:
            best, best_j = 0.0, -1
            for j, (gcls, gbox) in enumerate(gt):
                if used[j] or gcls != classes[index]:
                    continue
                value = float(iou_matrix(pred[index:index + 1], np.array([gbox], dtype=float))[0, 0])
                if value > best:
                    best, best_j = value, j
            if best_j >= 0 and best >= 0.5:
                used[best_j] = True
                tp[classes[index]] += 1
            else:
                fp[classes[index]] += 1
        for j, (gcls, _) in enumerate(gt):
            if not used[j]:
                fn[gcls] += 1
    return tp, fp, fn


def summarise(tp, fp, fn):
    total_tp, total_fp, total_fn = int(tp.sum()), int(fp.sum()), int(fn.sum())
    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {'tp': total_tp, 'fp': total_fp, 'fn': total_fn,
            'precision': round(precision, 4), 'recall': round(recall, 4), 'f1': round(f1, 4)}


def sweep(records, class_count):
    rows = []
    for threshold in THRESHOLDS:
        tp, fp, fn = evaluate(records, threshold, class_count)
        row = {'threshold': threshold, **summarise(tp, fp, fn),
               'per_class': {i: summarise(tp[i:i + 1], fp[i:i + 1], fn[i:i + 1]) for i in range(class_count)}}
        rows.append(row)
    return rows


def pick(rows):
    best_f1 = max(rows, key=lambda r: (r['f1'], r['recall']))
    precise = [r for r in rows if r['precision'] >= 0.70]
    best_precise = max(precise, key=lambda r: r['recall']) if precise else None
    return best_f1, best_precise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, help='training_runs/<run>/weights/best.pt')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--split', default='validation')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--output', required=True)
    parser.add_argument('--confirm-split', default='test')
    args = parser.parse_args()

    dataset = BASE / args.dataset
    weights = BASE / 'training_runs' / args.run / 'weights' / 'best.pt'
    model = YOLO(str(weights))
    names = [model.names[i] for i in sorted(model.names)]
    records = collect(model, dataset, args.split, args.imgsz)
    rows = sweep(records, len(names))
    best_f1, best_precise = pick(rows)
    payload = {
        'run': args.run, 'dataset': args.dataset, 'weights': str(weights), 'imgsz': args.imgsz,
        'calibration_split': args.split, 'images': len(records), 'classes': names,
        'selection_rule': 'primary = best F1; alternative = highest recall with precision >= 0.70',
        'best_f1': best_f1, 'best_precision_at_least_0.70': best_precise,
        'sweep': rows,
    }
    if args.confirm_split:
        confirm = collect(model, dataset, args.confirm_split, args.imgsz)
        payload['confirmation'] = {'split': args.confirm_split, 'images': len(confirm),
                                   'at_best_f1_threshold': summarize_at(confirm, best_f1['threshold'], len(names)),
                                   'at_0.50': summarize_at(confirm, 0.50, len(names)),
                                   'at_0.25': summarize_at(confirm, 0.25, len(names)),
                                   'sweep': sweep(confirm, len(names))}
    (BASE / args.output).write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(json.dumps({'run': args.run, 'split': args.split, 'images': len(records),
                      'best_f1': {k: best_f1[k] for k in ('threshold', 'precision', 'recall', 'f1')},
                      'precise': {k: best_precise[k] for k in ('threshold', 'precision', 'recall', 'f1')} if best_precise else None,
                      'confirmation': payload.get('confirmation')}, ensure_ascii=False))


def summarize_at(records, threshold, class_count):
    tp, fp, fn = evaluate(records, threshold, class_count)
    return summarise(tp, fp, fn)


if __name__ == '__main__':
    main()
