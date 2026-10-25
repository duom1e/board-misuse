"""Counterfactual demo with the FoodSeg103 model: the same board under two invented policies.

Picks one board test image where the model sees meat and one where it sees vegetables, then
runs the rule engine twice on each (meat-only vs vegetable-only). Everything is an invented
policy on a public photo; nothing here is an observed violation.
"""
import csv
import json
import os
from pathlib import Path

from check_board_misuse import inspect_image

BASE = Path(__file__).resolve().parent
os.environ['YOLO_CONFIG_DIR'] = str(BASE.parent / 'work' / 'yolo-config')


def detect_candidates():
    from ultralytics import YOLO
    model = YOLO(str(json.loads((BASE / 'active_models.json').read_text())['foodseg']['weights']))
    found = {'meat': None, 'vegetable': None}
    for path in sorted((BASE / 'board_dataset' / 'images' / 'test').glob('*.jpg')):
        result = model.predict(source=str(path), conf=0.5, imgsz=640, device=0, verbose=False)[0]
        if result.boxes is None or not len(result.boxes):
            continue
        names = [model.names[int(c)] for c in result.boxes.cls.cpu().tolist()]
        for wanted in ('meat', 'vegetable'):
            if found[wanted] is None and wanted in names:
                found[wanted] = path
        if all(found.values()):
            break
    return found


def main():
    found = detect_candidates()
    summary = {'note': 'Invented policies on public board photos. Not a violation, not an accuracy benchmark.',
               'model': 'foodseg', 'cases': []}
    for wanted, path in found.items():
        if path is None:
            summary['cases'].append({'wanted': wanted, 'found': False})
            continue
        policies = [('meat', ['meat']), ('vegetable', ['vegetable'])]
        entry = {'wanted': wanted, 'image': str(path), 'results': {}}
        for label, allowed in policies:
            dest = BASE / 'misuse_results' / f'foodseg_demo_{path.stem}_{label}'
            config = {'demo_not_site_policy': True, 'food_model': 'foodseg',
                      'description': 'Invented policy on a public image; software illustration only.',
                      'all_detected_boards_allowed_foods': allowed,
                      'board_confidence': .5, 'food_confidence': .5, 'min_food_overlap': .8}
            if not path.exists():
                continue
            report = inspect_image(path, config, dest)
            entry['results'][label] = {'overall_status': report['overall_status'],
                                       'foods': [{'category': d['category'], 'confidence': round(d['confidence'], 3)}
                                                 for d in report['food_detections']],
                                       'boards': [{'id': c['board_id'], 'status': c['status']} for c in report['checks']],
                                       'output': str(dest)}
        with (BASE / 'board_dataset' / 'attribution.csv').open(encoding='utf-8-sig') as handle:
            attribution = next((row for row in csv.DictReader(handle) if row['ImageID'] == path.stem), None)
        if attribution:
            attribution['Derivative'] = 'evidence.jpg: original image plus model predictions'
            for label, _ in policies:
                dest = BASE / 'misuse_results' / f'foodseg_demo_{path.stem}_{label}'
                (dest / 'attribution.json').write_text(json.dumps(attribution, ensure_ascii=False, indent=2))
        summary['cases'].append(entry)
        print(wanted, path.name, json.dumps({k: v['overall_status'] for k, v in entry['results'].items()}), flush=True)
    (BASE / 'foodseg_demo_compare.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print('WROTE foodseg_demo_compare.json')


if __name__ == '__main__':
    main()
