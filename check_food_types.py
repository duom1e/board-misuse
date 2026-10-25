"""Qualitative check: what the two-class food model says on a handful of board images.

This is a diagnostic sheet, not an accuracy measurement. It reuses the five board images
that carry no culinary vegetables plus a few plant/animal test images, and prints
class + confidence so wrong labels are visible instead of hidden in an aggregate metric.
"""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
DATA = BASE / 'food_taxonomy_v1'
CANDIDATES = [
    # (label, path, expectation written before running)
    ('cheese_board_train', BASE / 'board_dataset' / 'images' / 'train' / '020fe9bf9144e53b.jpg', 'animal_food or nothing'),
    ('bread_board_train', BASE / 'board_dataset' / 'images' / 'train' / '01152f92ce79f3d4.jpg', 'nothing (bread is not a class)'),
    ('loaves_board_train', BASE / 'board_dataset' / 'images' / 'train' / '0178460fd7fe47ea.jpg', 'nothing (bread is not a class)'),
    ('cello_workshop_board', BASE / 'board_dataset' / 'images' / 'train' / '0266b583b7f5ed35.jpg', 'nothing (workshop board)'),
    ('busses_board_train', BASE / 'board_dataset' / 'images' / 'train' / '013bc45321bef9c3.jpg', 'nothing'),
]
CONF = 0.25


def main(weights=None, dataset='food_taxonomy_v1', output=None):
    from ultralytics import YOLO
    data = BASE / dataset
    registry = json.loads((BASE / 'active_models.json').read_text())
    entry = dict(registry['food_type'])
    if weights:
        entry['weights'] = weights
    model = YOLO(entry['weights'])
    names = list(model.names.values())
    in_train = {p.stem for p in (data / 'images' / 'train').glob('*.jpg')} if (data / 'images' / 'train').exists() else set()
    rows = []
    for label, path, expectation in CANDIDATES:
        if not path.exists():
            rows.append({'case': label, 'path': str(path), 'missing': True})
            continue
        result = model.predict(source=str(path), conf=0.01, imgsz=int(entry.get('imgsz', 640)),
                               device=0, verbose=False)[0]
        detections = []
        if result.boxes is not None and len(result.boxes):
            for box, score, cls in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(),
                                       result.boxes.cls.cpu().tolist()):
                detections.append({'category': names[int(cls)], 'confidence': round(float(score), 4),
                                   'xyxy': [round(v, 1) for v in box]})
        detections.sort(key=lambda d: -d['confidence'])
        rows.append({'case': label, 'path': str(path), 'imgsz': int(entry.get('imgsz', 640)),
                     'expected': expectation, 'detections_at_0.01': detections,
                     'above_conf_0.25': [d for d in detections if d['confidence'] >= CONF],
                     'in_training_set': path.stem in in_train})
    payload = {'model': entry, 'dataset': dataset, 'confidence_floor_for_listing': CONF,
               'note': 'Curated examples, some inside the training set. Not a false-positive rate.',
               'rows': rows}
    destination = BASE / (output or 'food_type_qualitative.json')
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    summary = {r['case']: [(d['category'], d['confidence']) for d in r.get('above_conf_0.25', [])] for r in rows}
    print(json.dumps({'output': str(destination), 'summary': summary}, ensure_ascii=False))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default=None)
    parser.add_argument('--dataset', default='food_taxonomy_v1')
    parser.add_argument('--output', default=None)
    args = parser.parse_args()
    main(args.weights, args.dataset, args.output)
