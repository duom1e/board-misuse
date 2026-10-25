"""Compare two food-taxonomy runs on the same test split and the same held-out probe."""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent


def load(name):
    return json.loads((BASE / name).read_text())


def pick(payload):
    counts = payload['fixed_confidence_counts']
    probe = payload['board_only_negative_probe']
    return {
        'run': payload.get('run', Path(payload['dataset']).name),
        'dataset': payload['dataset'],
        'weights': payload['weights'],
        'test_mAP50': payload['framework_test_metrics'].get('metrics/mAP50(B)'),
        'test_mAP50_95': payload['framework_test_metrics'].get('metrics/mAP50-95(B)'),
        'counts_0.25': counts['0.25']['micro'],
        'counts_0.5': counts['0.5']['micro'],
        'per_class_0.5': counts['0.5']['per_class'],
        'probe_images_with_detection_0.25': probe['0.25']['images_with_any_detection'],
        'probe_images_with_detection_0.5': probe['0.5']['images_with_any_detection'],
        'probe_images': probe['0.5']['images'],
    }


def main():
    v1 = pick(load('food_taxonomy_metrics.json'))
    v2 = pick(load('food_taxonomy_v2_metrics.json'))
    comparison = {
        'v1': v1, 'v2': v2,
        'delta': {
            'test_mAP50': round(v2['test_mAP50'] - v1['test_mAP50'], 4),
            'test_mAP50_95': round(v2['test_mAP50_95'] - v1['test_mAP50_95'], 4),
            'recall_at_0.5': round(v2['counts_0.5']['recall'] - v1['counts_0.5']['recall'], 4),
            'precision_at_0.5': round(v2['counts_0.5']['precision'] - v1['counts_0.5']['precision'], 4),
            'probe_fp_images_at_0.5': v2['probe_images_with_detection_0.5'] - v1['probe_images_with_detection_0.5'],
        },
        'test_split_note': 'Both runs evaluate the identical 220-image test split; v2 only changed the training split.',
        'probe_note': 'The 22 board-only probe images are held out in both runs.',
    }
    (BASE / 'food_taxonomy_comparison.json').write_text(json.dumps(comparison, indent=2, ensure_ascii=False))
    print(json.dumps({'v1': {k: v1[k] for k in ('test_mAP50', 'test_mAP50_95', 'probe_images_with_detection_0.5')},
                      'v2': {k: v2[k] for k in ('test_mAP50', 'test_mAP50_95', 'probe_images_with_detection_0.5')},
                      'delta': comparison['delta']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
