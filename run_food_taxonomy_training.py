"""Train and evaluate the two-class food taxonomy detector (plant_food vs animal_food).

Adds two things the earlier vegetable experiments lacked:
  1. A second, explicitly different food class, so "wrong food on this board" is a real
     comparison instead of "vegetable vs nothing".
  2. Fixed-confidence counting metrics (TP/FP/FN at conf 0.25 and 0.50) plus a weak-label
     false-positive probe on cutting-board images that carry no food annotation at all.
"""
import json
import time
from pathlib import Path

import numpy as np
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent
DATA = BASE / 'food_taxonomy_v1'
PRETRAINED = BASE / 'yolo11n.pt'
RUN_NAME = 'food_taxonomy_v1'
RUNS = BASE / 'training_runs'
CLASSES = ['plant_food', 'animal_food']
CONF_LEVELS = [0.25, 0.50]


def read_classes(dataset_dir):
    """Class names from data.yaml, so 2-class and 3-class datasets share this script."""
    import yaml
    data = yaml.safe_load((dataset_dir / 'data.yaml').read_text())
    names = data['names']
    if isinstance(names, dict):
        return [names[i] for i in sorted(names)]
    return list(names)


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
            c, x, y, w, h = line.split()
            x, y, w, h = (float(v) for v in (x, y, w, h))
            rows.append((int(c), ((x - w / 2) * width, (y - h / 2) * height,
                                  (x + w / 2) * width, (y + h / 2) * height)))
    return rows


def count_metrics(model, split='test', imgsz=640):
    images = [Path(p) for p in (DATA / f'{split}.txt').read_text().split() if p.strip()]
    cache = {}
    for conf in CONF_LEVELS:
        cache[conf] = {c: {'tp': 0, 'fp': 0, 'fn': 0} for c in range(len(CLASSES))}
        cache[conf]['images_with_false_positive'] = 0
        cache[conf]['images'] = 0
    per_image = []
    for image_path in images:
        result = model.predict(source=str(image_path), conf=0.05, imgsz=imgsz, verbose=False)[0]
        width, height = result.orig_shape[1], result.orig_shape[0]
        gt = load_gt(DATA / 'labels' / split / f'{image_path.stem}.txt', width, height)
        pred = result.boxes.xyxy.cpu().numpy() if result.boxes is not None and len(result.boxes) else np.zeros((0, 4))
        scores = result.boxes.conf.cpu().numpy() if len(pred) else np.zeros(0)
        cls = result.boxes.cls.cpu().numpy().astype(int) if len(pred) else np.zeros(0, dtype=int)
        row = {'image': image_path.name, 'gt_boxes': len(gt), 'preds': {}}
        for conf in CONF_LEVELS:
            keep = scores >= conf
            p_boxes, p_scores, p_cls = pred[keep], scores[keep], cls[keep]
            gt_used = [False] * len(gt)
            order = np.argsort(-p_scores)
            matched = 0
            false_positives = 0
            for i in order:
                best_iou, best_j = 0.0, -1
                for j, (gcls, gbox) in enumerate(gt):
                    if gt_used[j] or gcls != p_cls[i]:
                        continue
                    value = float(iou_matrix(p_boxes[i:i + 1], np.array([gbox], dtype=float))[0, 0])
                    if value > best_iou:
                        best_iou, best_j = value, j
                if best_j >= 0 and best_iou >= 0.5:
                    gt_used[best_j] = True
                    matched += 1
                    cache[conf][int(p_cls[i])]['tp'] += 1
                else:
                    false_positives += 1
                    cache[conf][int(p_cls[i])]['fp'] += 1
            for j, (gcls, _) in enumerate(gt):
                if not gt_used[j]:
                    cache[conf][gcls]['fn'] += 1
            cache[conf]['images'] += 1
            if false_positives:
                cache[conf]['images_with_false_positive'] += 1
            row['preds'][conf] = {'detections': int(len(p_boxes)), 'matched': matched,
                                  'false_positives': false_positives}
        per_image.append(row)
    summary = {}
    for conf in CONF_LEVELS:
        block = cache[conf]
        total = {'tp': 0, 'fp': 0, 'fn': 0}
        per_class = {}
        for index, name in enumerate(CLASSES):
            stats = block[index]
            total['tp'] += stats['tp']
            total['fp'] += stats['fp']
            total['fn'] += stats['fn']
            denom_p = stats['tp'] + stats['fp']
            denom_r = stats['tp'] + stats['fn']
            per_class[name] = {**stats,
                               'precision': stats['tp'] / denom_p if denom_p else None,
                               'recall': stats['tp'] / denom_r if denom_r else None}
        summary[str(conf)] = {
            'per_class': per_class,
            'micro': {**total,
                      'precision': total['tp'] / (total['tp'] + total['fp']) if total['tp'] + total['fp'] else None,
                      'recall': total['tp'] / (total['tp'] + total['fn']) if total['tp'] + total['fn'] else None},
            'images': block['images'],
            'images_with_false_positive': block['images_with_false_positive']}
    return summary, per_image


def negative_probe(model, imgsz=640):
    manifest = json.loads((DATA / 'negatives_board.json').read_text())
    paths = [DATA / 'negatives_board' / f"{rec['image_id']}.jpg" for rec in manifest['images']]
    paths = [p for p in paths if p.exists()]
    probe = {str(c): {'images': len(paths), 'images_with_any_detection': 0, 'detections': 0,
                      'examples': []} for c in CONF_LEVELS}
    for path in paths:
        result = model.predict(source=str(path), conf=0.05, imgsz=imgsz, verbose=False)[0]
        for conf in CONF_LEVELS:
            if result.boxes is None or not len(result.boxes):
                continue
            scores = result.boxes.conf.cpu().numpy()
            muted = scores >= conf
            if muted.any():
                probe[str(conf)]['images_with_any_detection'] += 1
                probe[str(conf)]['detections'] += int(muted.sum())
                probe[str(conf)]['examples'].append({'image': path.name,
                                                     'confidences': [round(float(s), 4) for s in scores[muted]]})
    probe['note'] = ('Cutting-board images with no food annotation in Open Images. This is a weak-label '
                     'probe: unannotated food can still be present, so the rate is neither a precision nor '
                     'a real-kitchen false-alarm rate.')
    return probe


def main(dataset='food_taxonomy_v1', run_name='food_taxonomy_v1', weights_file='yolo11n.pt', epochs=30):
    global DATA, RUN_NAME, CLASSES
    DATA = BASE / dataset
    RUN_NAME = run_name
    CLASSES = read_classes(DATA)
    started = time.time()
    model = YOLO(str(BASE / weights_file))
    model.train(data=str(DATA / 'data.yaml'), epochs=epochs, imgsz=640, batch=8, device=0, workers=0,
                amp=False, seed=42, deterministic=True, project=str(RUNS), name=RUN_NAME,
                exist_ok=True, patience=0, val=True, plots=False, verbose=True)
    weights = RUNS / RUN_NAME / 'weights' / 'best.pt'
    trained = YOLO(str(weights))
    val = trained.val(data=str(DATA / 'data.yaml'), split='test', imgsz=640, batch=8, device=0,
                      workers=0, project=str(RUNS), name=f'{RUN_NAME}_test', exist_ok=True, plots=False)
    metrics = {k: float(v) for k, v in dict(val.results_dict).items()}
    counts, per_image = count_metrics(trained)
    probe = negative_probe(trained)
    payload = {
        'dataset': str(DATA),
        'classes': CLASSES,
        'weights': str(weights),
        'start_weights': str(BASE / weights_file),
        'epochs': epochs,
        'imgsz': 640,
        'framework_test_metrics': metrics,
        'fixed_confidence_counts': counts,
        'per_image': per_image,
        'board_only_negative_probe': probe,
        'elapsed_seconds': round(time.time() - started, 1),
        'limitations': ['Food type classes only; no raw/cooked (生熟) capability.',
                        'Open Images boxes are crowd annotations, not kitchen-safety expert labels.',
                        'Board-only negatives are weak labels.'],
    }
    (BASE / f'{RUN_NAME}_metrics.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print('FOOD_TRAINING_COMPLETE', json.dumps({
        'run': RUN_NAME,
        'test_metrics': metrics,
        'counts_at_0.25': counts['0.25']['micro'],
        'counts_at_0.5': counts['0.5']['micro'],
        'negative_probe': {k: probe[k] for k in ('0.25', '0.5')},
    }, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='food_taxonomy_v1')
    parser.add_argument('--run-name', default=None)
    parser.add_argument('--weights', default='yolo11n.pt')
    parser.add_argument('--epochs', type=int, default=30)
    args = parser.parse_args()
    main(args.dataset, args.run_name or args.dataset, args.weights, args.epochs)
