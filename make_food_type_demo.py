"""Side-by-side demo: legacy vegetable model vs the two-class food_type model.

Two cases only:
  1. A cheese board (no culinary vegetables). Declared policy: animal_food only.
     The legacy model calls the cheese "vegetable" -> false alarm under a meat/animal policy.
     The new model can call it animal_food -> no conflict.
  2. A bread board. Declared policy: plant_food only. Bread is not a class, so both models
     should stay quiet; if either fires, that is a visible false positive.

Every folder also records that the policy is invented for illustration. These are curated
examples, not a false-alarm-rate measurement.
"""
import csv
import json
import os
from pathlib import Path

from check_board_misuse import inspect_image

BASE = Path(__file__).resolve().parent
os.environ['YOLO_CONFIG_DIR'] = str(BASE.parent / 'work' / 'yolo-config')

CASES = [
    {'name': 'cheese_animal_only', 'image': '020fe9bf9144e53b', 'split': 'train',
     'allowed': ['animal_food'], 'why': 'cheese is animal-derived food, board declared animal_food only'},
    {'name': 'cheese_plant_only', 'image': '020fe9bf9144e53b', 'split': 'train',
     'allowed': ['plant_food'], 'why': 'same image under a plant-only policy'},
    {'name': 'bread_plant_only', 'image': '01152f92ce79f3d4', 'split': 'train',
     'allowed': ['plant_food'], 'why': 'bread is not one of the modelled classes'},
]
# The legacy model has no animal/plant words, so the same policy is expressed in its vocabulary.
LEGACY_WORD = {'animal_food': 'meat', 'plant_food': 'vegetable'}


def run():
    summary = {'note': 'Curated illustration images, some inside the training split. Not an accuracy benchmark.',
               'cases': []}
    for case in CASES:
        source = BASE / 'board_dataset' / 'images' / case['split'] / f"{case['image']}.jpg"
        if not source.exists():
            summary['cases'].append({**case, 'missing': True})
            continue
        entry = {'case': case['name'], 'image': str(source), 'policy': case['allowed'], 'why': case['why'], 'models': {}}
        for model_key in ['vegetable', 'food_type']:
            dest = BASE / 'misuse_results' / f"food_type_demo_{model_key}_{case['name']}"
            policy = [LEGACY_WORD.get(a, a) if model_key == 'vegetable' else a for a in case['allowed']]
            config = {'demo_not_site_policy': True, 'food_model': model_key,
                      'description': 'Invented policy on a public image; software illustration only.',
                      'all_detected_boards_allowed_foods': policy,
                      'board_confidence': .5, 'food_confidence': .5, 'min_food_overlap': .8}
            try:
                report = inspect_image(source, config, dest)
            except RuntimeError as exc:
                entry['models'][model_key] = {'error': str(exc)}
                continue
            entry['models'][model_key] = {
                'policy_words_used': policy,
                'overall_status': report['overall_status'],
                'supported_food_categories': report['supported_food_categories'],
                'detections': [{'category': d['category'], 'confidence': round(d['confidence'], 4)}
                               for d in report['food_detections']],
                'boards': [{'id': c['board_id'], 'status': c['status'],
                            'conflicts': [e for e in c['evidence'] if e['conflicts_with_rule']]}
                           for c in report['checks']],
                'output': str(dest),
            }
        summary['cases'].append(entry)
        print(case['name'], json.dumps({k: v.get('overall_status') for k, v in entry['models'].items()}), flush=True)
    (BASE / 'food_type_demo_compare.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print('WROTE food_type_demo_compare.json', flush=True)


if __name__ == '__main__':
    run()
