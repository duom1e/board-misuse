"""Visual verification of the fine-tuned board model on the held-out kitchen images."""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
PROBE = BASE / 'real_kitchen_probe'

# Tile order matches board_holdout_after.jpg (first 12 flagged hold-out images).
VERDICTS = [
    {'image': '8638f05fa3765b3a', 'boxes': 1, 'board_boxes': 0, 'seen': 'coffee cup on a saucer'},
    {'image': '8fdb919566b4ba09', 'boxes': 1, 'board_boxes': 1, 'seen': 'knife resting on a light wooden board'},
    {'image': '91cdbbf781c463ff', 'boxes': 1, 'board_boxes': 0, 'seen': 'bowl of dough with a scraper'},
    {'image': 'a2da9215e6d732c8', 'boxes': 1, 'board_boxes': 0, 'seen': 'stainless bowl with a whisk'},
    {'image': 'bafd801621510353', 'boxes': 1, 'board_boxes': 0, 'seen': 'hand holding a knife against a dark table'},
    {'image': 'bc857cdddd5c152e', 'boxes': 3, 'board_boxes': 2, 'seen': 'knife and coins on a wooden board'},
    {'image': 'bf0ad1c2a08e14b3', 'boxes': 2, 'board_boxes': 2, 'seen': 'leeks and a knife on a blue board'},
    {'image': 'c0ad154a8b0248ac', 'boxes': 1, 'board_boxes': 0, 'seen': 'spatula held up against a plain wall'},
    {'image': 'db951e5b54750a7a', 'boxes': 1, 'board_boxes': 0, 'seen': 'lemons, jar and spoons on a tray'},
    {'image': 'e1f2937a9d457b15', 'boxes': 2, 'board_boxes': 2, 'seen': 'knife on a wooden board'},
    {'image': 'e4746e7f86c136e1', 'boxes': 1, 'board_boxes': 0, 'seen': 'plate of dumplings'},
    {'image': 'e7b58bd9c1239800', 'boxes': 1, 'board_boxes': 0, 'seen': 'wooden bowl of rice'},
]


def main():
    metrics = json.loads((BASE / 'board_v2_metrics.json').read_text())
    baseline_precision = 0.087  # from verification.json on the same probe, pre-tuning sample
    boxes = sum(row['boxes'] for row in VERDICTS)
    correct = sum(row['board_boxes'] for row in VERDICTS)
    payload = {
        'model': 'training_runs/board_v2_real/weights/best.pt',
        'holdout': 'the 92 kitchen images not used for labelling or training',
        'verified_images': len(VERDICTS),
        'boxes_seen': boxes,
        'board_boxes_correct': correct,
        'precision_among_detections': round(correct / boxes, 3) if boxes else None,
        'baseline_precision_same_probe_before_tuning': baseline_precision,
        'detection_counts': {'before': metrics['kitchen_holdout_before'], 'after': metrics['kitchen_holdout_after']},
        'board_dataset_test': {'before': metrics['board_dataset_test_before'], 'after': metrics['board_dataset_test_after']},
        'finding': 'Fine-tuning on 2 verified real boards and 44 verified no-board scenes cut detections on held-out kitchen '
                   'images from 29 to 17 boxes and lifted precision among detections from about 0.09 to about 0.44.',
        'limitations': [
            'Only 12 of the 13 flagged held-out images were checked (one image with a single box is unverified).',
            'Single-rater assistant screening, not expert labelling.',
            'This is precision among detections; recall is unmeasured because the hold-out has no boxes.',
            'The hold-out is only 92 images and shares the Open Images photo style with the training additions.',
        ],
        'rows': VERDICTS,
    }
    (PROBE / 'board_v2_verification.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(json.dumps({k: payload[k] for k in ('verified_images', 'boxes_seen', 'board_boxes_correct',
                                              'precision_among_detections', 'detection_counts')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
