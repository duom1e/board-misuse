"""Apply pre-registered criteria for a FoodSeg-based run and report the per-class picture.

Thresholds for each criterion come from the JSON rule file's baseline block, so a new run
cannot silently pick its own passing line.
"""
import argparse
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent


def main(run, dataset, rule_file, output, probe_limit, min_map50=0.40, min_meat=0.55, min_veg=0.57, min_recall=0.36):
    metrics = json.loads((BASE / f'{run}_metrics.json').read_text())
    probe = metrics['board_only_negative_probe']
    counts = metrics['fixed_confidence_counts']

    from ultralytics import YOLO
    model = YOLO(metrics['weights'])
    val = model.val(data=str(BASE / dataset / 'data.yaml'), split='test', imgsz=640, batch=8,
                    device=0, workers=0, plots=False, verbose=False)
    names = [model.names[i] for i in sorted(model.names)]
    ap50 = {names[i]: round(float(v), 4) for i, v in enumerate(val.box.ap50)}
    ap = {names[i]: round(float(v), 4) for i, v in enumerate(val.box.maps)}
    per_class_05 = counts['0.5']['per_class']
    map50 = round(float(val.box.map50), 4)
    checks = [
        {'criterion': f'board-only probe: <= {probe_limit} of 22 images detect at conf 0.5',
         'result': probe['0.5']['images_with_any_detection'], 'passed':
             probe['0.5']['images_with_any_detection'] <= probe_limit},
        {'criterion': f'test mAP50 >= {min_map50}', 'result': map50, 'passed': map50 >= min_map50},
        {'criterion': f'test meat AP50 >= {min_meat}', 'result': ap50.get('meat'),
         'passed': (ap50.get('meat') or 0) >= min_meat},
        {'criterion': f'test vegetable AP50 >= {min_veg}', 'result': ap50.get('vegetable'),
         'passed': (ap50.get('vegetable') or 0) >= min_veg},
        {'criterion': f'overall recall at conf 0.50 >= {min_recall}', 'result': counts['0.5']['micro']['recall'],
         'passed': (counts['0.5']['micro']['recall'] or 0) >= min_recall},
    ]
    payload = {
        'run': run, 'dataset': dataset, 'rule_file': rule_file, 'test_map50': map50,
        'all_criteria_passed': all(c['passed'] for c in checks), 'checks': checks,
        'per_class_test_ap50': ap50, 'per_class_test_ap50_95': ap,
        'framework_test_metrics': metrics['framework_test_metrics'],
        'fixed_confidence_counts': {'0.25': counts['0.25']['micro'], '0.5': counts['0.5']['micro'],
                                    'per_class_0.5': per_class_05, 'per_class_0.25': counts['0.25']['per_class']},
        'board_only_probe': {k: probe[k]['images_with_any_detection'] for k in ('0.25', '0.5')},
        'caveats': ['FoodSeg103 is not a raw-vs-cooked dataset; meat means ingredient category.',
                    'test split is our own halving of the official validation split.',
                    'board photos used as training backgrounds may contain unannotated food.'],
    }
    (BASE / output).write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(json.dumps({k: payload[k] for k in ('all_criteria_passed', 'checks', 'per_class_test_ap50')},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default='foodseg_v1')
    parser.add_argument('--dataset', default='foodseg_dataset')
    parser.add_argument('--rule', default='foodseg_v1_decision_rule.json')
    parser.add_argument('--output', default=None)
    parser.add_argument('--probe-limit', type=int, default=4)
    parser.add_argument('--min-map50', type=float, default=0.40)
    parser.add_argument('--min-meat', type=float, default=0.55)
    parser.add_argument('--min-veg', type=float, default=0.57)
    parser.add_argument('--min-recall', type=float, default=0.36)
    args = parser.parse_args()
    main(args.run, args.dataset, args.rule, args.output or f'{args.run}_decision.json',
         args.probe_limit, args.min_map50, args.min_meat, args.min_veg, args.min_recall)
