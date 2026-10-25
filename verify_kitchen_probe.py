"""Combine the kitchen probe detections with a visual verdict sheet and compute precision.

Verdicts were made by looking at the contact sheets (24 images, the first 24 flagged ones in
deterministic order). Each entry lists how many boxes on that image were correct. This is
assistant screening, single rater, and the sample is conditional on having a detection, so the
result is 'precision among detections', not a rate over all kitchen images.
"""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
PROBE = BASE / 'real_kitchen_probe'

# boxes_correct counts how many of the drawn boxes were actually on a board / on food.
VERDICTS = {
    '03470d04c056': {'board_correct': 0, 'meat_correct': 0, 'veg_correct': 0, 'seen': 'wooden bowls on a shelf, no cutting board'},
    '0aa532f51ab0': {'board_correct': 0, 'meat_correct': 0, 'veg_correct': 0, 'seen': 'soup bowl, no cutting board'},
    '0ab25a872a3a': {'board_correct': 0, 'meat_correct': 0, 'veg_correct': 0, 'seen': 'forks on a plate'},
    '0aec86633421': {'board_correct': 0, 'meat_correct': 0, 'veg_correct': 0, 'seen': 'coffee cup on a table'},
    '0b6a55f2f634': {'board_correct': 0, 'meat_correct': 0, 'veg_correct': 0, 'seen': 'whisk in a mixing bowl'},
    '10636103fb39': {'board_correct': 0, 'meat_correct': 2, 'veg_correct': 0, 'seen': 'sausages on a grill'},
    '16207d46f3b9': {'board_correct': 0, 'meat_correct': 0, 'veg_correct': 0, 'seen': 'knife on a plain grey surface'},
    '1b16cb29d5fb': {'board_correct': 0, 'meat_correct': 0, 'veg_correct': 0, 'seen': 'empty kitchen with cabinets'},
    '1c2b0838f50a': {'board_correct': 0, 'meat_correct': 0, 'veg_correct': 0, 'seen': 'knife on a mat'},
    '1c6e8f708bc4': {'board_correct': 0, 'meat_correct': 0, 'veg_correct': 0, 'seen': 'noodle soup bowl'},
    '1de9f2111020': {'board_correct': 0, 'meat_correct': 1, 'veg_correct': 0, 'seen': 'plate with egg and meat'},
    '2167755de02e': {'board_correct': 0, 'meat_correct': 0, 'veg_correct': 0, 'seen': 'two knives on a plain surface'},
    '3190886f9fe4': {'board_correct': 0, 'meat_correct': 0, 'veg_correct': 0, 'seen': 'thermos and printer'},
    '3254bc540ca2': {'board_correct': 0, 'meat_correct': 0, 'veg_correct': 0, 'seen': 'knife on a woven mat'},
    '3337c010f45c': {'board_correct': 0, 'meat_correct': 0, 'veg_correct': 0, 'seen': 'stand mixer'},
    '34c6b04d6357': {'board_correct': 0, 'meat_correct': 2, 'veg_correct': 0, 'seen': 'burgers and sausages on a grill'},
    '3af2bd1dd1f8': {'board_correct': 0, 'meat_correct': 0, 'veg_correct': 0, 'seen': 'people at a stove, no meat visible'},
    '3b23c9d71275': {'board_correct': 1, 'meat_correct': 0, 'veg_correct': 0, 'seen': 'knife lying on a wooden cutting board'},
    '46168164054d': {'board_correct': 0, 'meat_correct': 1, 'veg_correct': 0, 'seen': 'grill basket full of meat'},
    '4727f2740dc4': {'board_correct': 0, 'meat_correct': 0, 'veg_correct': 0, 'seen': 'green soup plate'},
    '493fc196e970': {'board_correct': 1, 'meat_correct': 0, 'veg_correct': 0, 'seen': 'three knives on a wooden cutting board'},
    '4d78cffd3eae': {'board_correct': 0, 'meat_correct': 0, 'veg_correct': 0, 'seen': 'ceramic pitcher on a tray'},
    '5242f019fe38': {'board_correct': 0, 'meat_correct': 1, 'veg_correct': 0, 'seen': 'casserole of pulled meat'},
    '5b49d5465d93': {'board_correct': 0, 'meat_correct': 0, 'veg_correct': 0, 'seen': 'washing machine'},
}


def main():
    probe = json.loads((PROBE / 'domain_probe.json').read_text())
    flagged = [r for r in probe['images'] if r['boards_at_0.5'] or r['foods_at_0.5']]
    sample = flagged[:24]
    rows, totals = [], {'board': [0, 0], 'meat': [0, 0], 'veg': [0, 0]}
    for record in sample:
        stem = record['image'].replace('.jpg', '')
        verdict = VERDICTS.get(stem) or next((v for k, v in VERDICTS.items() if stem.startswith(k)), None)
        if verdict is None:
            continue
        boards = len(record['boards_at_0.5'])
        meat = sum(1 for d in record['foods_at_0.5'] if d['category'] == 'meat')
        veg = sum(1 for d in record['foods_at_0.5'] if d['category'] == 'vegetable')
        rows.append({'image': record['image'], 'board_boxes': boards, 'board_correct': verdict['board_correct'],
                     'meat_boxes': meat, 'meat_correct': verdict['meat_correct'],
                     'veg_boxes': veg, 'veg_correct': verdict['veg_correct'], 'seen': verdict['seen']})
        totals['board'][0] += boards
        totals['board'][1] += verdict['board_correct']
        totals['meat'][0] += meat
        totals['meat'][1] += verdict['meat_correct']
        totals['veg'][0] += veg
        totals['veg'][1] += verdict['veg_correct']
    food_boxes = totals['meat'][0] + totals['veg'][0]
    food_correct = totals['meat'][1] + totals['veg'][1]
    payload = {
        'probe': 'real_kitchen_probe (140 Open Images kitchen scenes, val/test only, never trained on)',
        'images_total': probe['summary']['images'],
        'images_with_board_at_0.5': probe['summary']['images_with_board_at_0.5'],
        'images_with_food_at_0.5': probe['summary']['images_with_food_at_0.5'],
        'sampled_images': len(rows),
        'sampling': 'first 24 flagged images in deterministic order; conditional on having a detection',
        'precision_among_detections': {
            'board': {'boxes': totals['board'][0], 'correct': totals['board'][1],
                      'precision': round(totals['board'][1] / totals['board'][0], 3) if totals['board'][0] else None},
            'meat': {'boxes': totals['meat'][0], 'correct': totals['meat'][1],
                     'precision': round(totals['meat'][1] / totals['meat'][0], 3) if totals['meat'][0] else None},
            'vegetable': {'boxes': totals['veg'][0], 'correct': totals['veg'][1],
                          'precision': round(totals['veg'][1] / totals['veg'][0], 3) if totals['veg'][0] else None},
            'food_overall': {'boxes': food_boxes, 'correct': food_correct,
                             'precision': round(food_correct / food_boxes, 3) if food_boxes else None},
        },
        'finding': 'On real kitchen scenes the board detector is the weak link: it fires on bowls, forked plates, '
                   'mixers and washing machines. Food detections, when they happen, are mostly correct.',
        'limitations': [
            'Single-rater assistant screening of contact sheets, not expert labelling.',
            'Sample is conditional on detections at conf 0.5, so this is precision, not recall.',
            'Only 24 of 61 flagged images were checked; the other 37 are unverified.',
            'Kitchen scenes are Open Images photos (mostly home kitchens), not commercial canteens.',
        ],
        'rows': rows,
    }
    (PROBE / 'verification.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(json.dumps({k: payload[k] for k in ('images_total', 'sampled_images', 'precision_among_detections')},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
