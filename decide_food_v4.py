"""Apply the pre-registered v4 criteria (YOLO11s on the v3 data) to the results."""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
RUN = 'food_taxonomy_v4s'
DATASET = 'food_taxonomy_v3'


def main():
    metrics = json.loads((BASE / f'{RUN}_metrics.json').read_text())
    qualitative = json.loads((BASE / 'food_type_qualitative_v4s.json').read_text())
    cheese = json.loads((BASE / 'cheese_recall_comparison_v4s.json').read_text())
    rows = {row['case']: row for row in qualitative['rows']}

    from ultralytics import YOLO
    model = YOLO(metrics['weights'])
    val = model.val(data=str(BASE / DATASET / 'data.yaml'), split='test', imgsz=640, batch=8,
                    device=0, workers=0, plots=False, verbose=False)
    names = [model.names[i] for i in sorted(model.names)]
    ap50 = {names[i]: round(float(v), 4) for i, v in enumerate(val.box.ap50)}
    counts = metrics['fixed_confidence_counts']
    recall_05 = counts['0.5']['micro']['recall']
    cheese_v4 = cheese['totals'].get('v4', {}).get('recall_proxy')
    cheese_note = None
    if cheese_v4 is None:
        cheese_note = 'cheese comparison was not run for this model; criterion cannot be evaluated'
    checks = [
        {'criterion': 'test animal_food AP50 >= 0.30', 'result': ap50.get('animal_food'),
         'passed': (ap50.get('animal_food') or 0) >= 0.30},
        {'criterion': 'test other_food AP50 >= 0.50', 'result': ap50.get('other_food'),
         'passed': (ap50.get('other_food') or 0) >= 0.50},
        {'criterion': 'cheese matched recall at conf 0.5 >= 0.11',
         'result': cheese_v4, 'passed': (cheese_v4 or 0) >= 0.11, 'note': cheese_note},
        {'criterion': 'overall recall at fixed conf 0.5 >= 0.11', 'result': round(recall_05, 4),
         'passed': recall_05 >= 0.11},
    ]
    payload = {
        'run': RUN, 'dataset': DATASET, 'rule_file': 'food_taxonomy_v4_decision_rule.json',
        'all_criteria_passed': all(c['passed'] for c in checks),
        'checks': checks, 'per_class_test_ap50': ap50,
        'fixed_confidence_counts': {'0.25': counts['0.25']['micro'], '0.5': counts['0.5']['micro']},
        'cheese_totals': {k: v for k, v in cheese['totals'].items()},
        'qualitative_hits_0.5': {name: [d for d in row.get('detections_at_0.01', []) if d['confidence'] >= 0.5]
                                for name, row in rows.items()},
        'contaminated_diagnostics': [name for name, row in rows.items() if row.get('in_training_set')],
    }
    (BASE / f'{RUN}_decision.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(json.dumps({k: payload[k] for k in ('all_criteria_passed', 'checks', 'per_class_test_ap50')},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
