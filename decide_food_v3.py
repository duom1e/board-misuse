"""Apply the pre-registered v3 criteria (food_taxonomy_v3_decision_rule.json) to the results.

Everything is computed from files, so the decision is auditable:
  food_taxonomy_v3_metrics.json  (test metrics, fixed-confidence counts, probe)
  food_type_qualitative_v3.json  (curated diagnostics, with in_training_set flags)
"""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent


def main():
    rule = json.loads((BASE / 'food_taxonomy_v3_decision_rule.json').read_text())
    metrics = json.loads((BASE / 'food_taxonomy_v3_metrics.json').read_text())
    qualitative = json.loads((BASE / 'food_type_qualitative_v3.json').read_text())
    rows = {row['case']: row for row in qualitative['rows']}

    def detections(case, floor=0.5):
        return [d for d in rows[case].get('detections_at_0.01', []) if d['confidence'] >= floor]

    from ultralytics import YOLO
    model = YOLO(metrics['weights'])
    val = model.val(data=str(BASE / 'food_taxonomy_v3' / 'data.yaml'), split='test', imgsz=640,
                    batch=8, device=0, workers=0, plots=False, verbose=False)
    names = [model.names[i] for i in sorted(model.names)]
    ap50 = {names[i]: round(float(v), 4) for i, v in enumerate(val.box.ap50)}
    ap = {names[i]: round(float(v), 4) for i, v in enumerate(val.box.maps)}
    counts = metrics['fixed_confidence_counts']
    recall_05 = counts['0.5']['micro']['recall']

    loaves = [d for d in detections('loaves_board_train') if d['category'] == 'other_food']
    loaves_wrong = [d for d in detections('loaves_board_train') if d['category'] in ('plant_food', 'animal_food')]
    cheese = [d for d in detections('cheese_board_train') if d['category'] == 'animal_food']
    checks = [
        {'criterion': 'loaves (held out) gets other_food at conf >= 0.5',
         'result': round(loaves[0]['confidence'], 4) if loaves else None,
         'passed': bool(loaves)},
        {'criterion': 'loaves has no plant_food/animal_food at conf >= 0.5',
         'result': [{'category': d['category'], 'confidence': d['confidence']} for d in loaves_wrong],
         'passed': not loaves_wrong},
        {'criterion': 'cheese (held out) still gets animal_food at conf >= 0.5',
         'result': round(cheese[0]['confidence'], 4) if cheese else None,
         'passed': bool(cheese)},
        {'criterion': 'test other_food mAP50 >= 0.25',
         'result': ap50.get('other_food'),
         'passed': (ap50.get('other_food') or 0) >= 0.25},
        {'criterion': 'overall recall at fixed conf 0.50 >= 0.09',
         'result': round(recall_05, 4),
         'passed': recall_05 >= 0.09},
    ]
    payload = {
        'rule_file': 'food_taxonomy_v3_decision_rule.json',
        'all_criteria_passed': all(c['passed'] for c in checks),
        'checks': checks,
        'per_class_test_ap50': ap50,
        'per_class_test_ap50_95': ap,
        'fixed_confidence_counts': {'0.25': counts['0.25']['micro'], '0.5': counts['0.5']['micro']},
        'contaminated_diagnostics': [name for name, row in rows.items() if row.get('in_training_set')],
        'board_only_probe_context_only': {
            'images': metrics['board_only_negative_probe']['0.5']['images'],
            'with_detection_0.5': metrics['board_only_negative_probe']['0.5']['images_with_any_detection'],
            'note': 'These 22 images were used for the v1/v2 decision, so this is context, not a criterion.'},
    }
    (BASE / 'food_taxonomy_v3_decision.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(json.dumps({k: payload[k] for k in ('all_criteria_passed', 'checks', 'per_class_test_ap50')},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
