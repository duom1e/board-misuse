"""Apply criteria to a run, using the audited probe ground truth instead of raw labels.

Raw probe counts treat every detection on a board-only image as a false positive. The audit
(probe_audit.json) found that some of those images do contain food, so a detection there is
correct. This script reports both numbers and evaluates the criteria against the corrected
one, with the correction written into the output so nothing is hidden.
"""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent


def main(run, dataset, output, rule_file, min_map50, min_meat, min_veg, min_recall, probe_limit):
    metrics = json.loads((BASE / f'{run}_metrics.json').read_text())
    audit = json.loads((BASE / 'probe_audit.json').read_text())
    counts = metrics['fixed_confidence_counts']
    food_ids = {row['image_id'] for row in audit['verdicts'] if row['verdict'] == 'food_present'}

    from ultralytics import YOLO
    model = YOLO(metrics['weights'])
    val = model.val(data=str(BASE / dataset / 'data.yaml'), split='test', imgsz=640, batch=8,
                    device=0, workers=0, plots=False, verbose=False)
    names = [model.names[i] for i in sorted(model.names)]
    ap50 = {names[i]: round(float(v), 4) for i, v in enumerate(val.box.ap50)}
    ap = {names[i]: round(float(v), 4) for i, v in enumerate(val.box.maps)}
    map50 = round(float(val.box.map50), 4)

    raw_probe = metrics['board_only_negative_probe']
    raw_examples = {row['image'].replace('.jpg', '') for row in raw_probe['0.5']['examples']}
    corrected_hits = sorted(raw_examples - food_ids)
    payload_checks = [
        {'criterion': f'audited probe false positives at conf 0.5 <= {probe_limit}',
         'result': len(corrected_hits), 'raw_result': len(raw_examples),
         'excluded_correct_detections': sorted(raw_examples & food_ids), 'passed': len(corrected_hits) <= probe_limit},
        {'criterion': f'test mAP50 >= {min_map50}', 'result': map50, 'passed': map50 >= min_map50},
        {'criterion': f'test meat AP50 >= {min_meat}', 'result': ap50.get('meat'),
         'passed': (ap50.get('meat') or 0) >= min_meat},
        {'criterion': f'test vegetable AP50 >= {min_veg}', 'result': ap50.get('vegetable'),
         'passed': (ap50.get('vegetable') or 0) >= min_veg},
        {'criterion': f'overall recall at conf 0.50 >= {min_recall}',
         'result': counts['0.5']['micro']['recall'],
         'passed': (counts['0.5']['micro']['recall'] or 0) >= min_recall},
    ]
    payload = {
        'run': run, 'dataset': dataset, 'rule_file': rule_file, 'probe_audit': 'probe_audit.json',
        'all_criteria_passed': all(c['passed'] for c in payload_checks), 'checks': payload_checks,
        'test_map50': map50, 'per_class_test_ap50': ap50, 'per_class_test_ap50_95': ap,
        'framework_test_metrics': metrics['framework_test_metrics'],
        'fixed_confidence_counts': {'0.25': counts['0.25']['micro'], '0.5': counts['0.5']['micro'],
                                    'per_class_0.5': counts['0.5']['per_class']},
        'probe': {'raw_at_0.5': len(raw_examples), 'corrected_at_0.5': len(corrected_hits),
                  'images': raw_probe['0.5']['images']},
        'correction_applied': 'detections on probe images that the audit shows contain food are treated as correct',
        'caveats': ['FoodSeg103 is not a raw-vs-cooked dataset.',
                    'Test split is our own halving of the official validation split.',
                    'The probe audit was done after seeing scores; it corrects wrong labels, it does not tune.'],
    }
    (BASE / output).write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(json.dumps({k: payload[k] for k in ('all_criteria_passed', 'checks', 'test_map50', 'per_class_test_ap50')},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--rule', default='foodseg_two_class_decision_rule.json')
    parser.add_argument('--output', default=None)
    parser.add_argument('--min-map50', type=float, default=0.40)
    parser.add_argument('--min-meat', type=float, default=0.55)
    parser.add_argument('--min-veg', type=float, default=0.57)
    parser.add_argument('--min-recall', type=float, default=0.36)
    parser.add_argument('--probe-limit', type=int, default=2)
    args = parser.parse_args()
    main(args.run, args.dataset, args.output or f'{args.run}_decision.json', args.rule,
         args.min_map50, args.min_meat, args.min_veg, args.min_recall, args.probe_limit)
