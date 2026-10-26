"""Record the visual verdicts on the 24 flagged Pexels photos and compute precision."""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
PROBE = BASE / 'pexels_kitchen' / 'probe'

# board_correct / food_correct count how many of that image's boxes were right.
VERDICTS = {
    'pexels_10201775.jpg': {'board_correct': 0, 'food_correct': 2, 'seen': 'plated noodles and dumplings; meat and vegetables present'},
    'pexels_10821328.jpg': {'board_correct': 0, 'food_correct': 1, 'seen': 'two bowls of food on a dining table, no cutting board'},
    'pexels_11020455.jpg': {'board_correct': 0, 'food_correct': 0, 'seen': 'styled kitchen counter with ceramics'},
    'pexels_12303265.jpg': {'board_correct': 0, 'food_correct': 0, 'seen': 'dining table set with glasses'},
    'pexels_13426497.jpg': {'board_correct': 0, 'food_correct': 0, 'seen': 'street and building facade'},
    'pexels_14987811.jpg': {'board_correct': 0, 'food_correct': 0, 'seen': 'dessert on a plate'},
    'pexels_1603899.jpg': {'board_correct': 0, 'food_correct': 1, 'seen': 'charcuterie board with fruit and bread, no visible meat cut'},
    'pexels_16177509.jpg': {'board_correct': 0, 'food_correct': 0, 'seen': 'street with buildings'},
    'pexels_17057004.jpg': {'board_correct': 0, 'food_correct': 0, 'seen': 'restaurant terrace dining area'},
    'pexels_17321330.jpg': {'board_correct': 0, 'food_correct': 0, 'seen': 'dessert in a cup'},
    'pexels_18978083.jpg': {'board_correct': 0, 'food_correct': 0, 'seen': 'cutlery laid on a placemat'},
    'pexels_19859287.jpg': {'board_correct': 0, 'food_correct': 0, 'seen': 'plated salad, no meat visible'},
    'pexels_20169774.jpg': {'board_correct': 0, 'food_correct': 1, 'seen': 'table with several dishes'},
    'pexels_26859038.jpg': {'board_correct': 0, 'food_correct': 0, 'seen': 'styled kitchen island'},
    'pexels_29219953.jpg': {'board_correct': 0, 'food_correct': 0, 'seen': 'restaurant interior'},
    'pexels_29460745.jpg': {'board_correct': 0, 'food_correct': 0, 'seen': 'cookies on a plate by a window'},
    'pexels_30650794.jpg': {'board_correct': 1, 'food_correct': 0, 'seen': 'cookie dough and cutters on a wooden board'},
    'pexels_34153651.jpg': {'board_correct': 0, 'food_correct': 0, 'seen': 'chocolate dessert'},
    'pexels_34274122.jpg': {'board_correct': 0, 'food_correct': 0, 'seen': 'bowl of cocoa powder'},
    'pexels_34434686.jpg': {'board_correct': 1, 'food_correct': 0, 'seen': 'sliced bananas on a wooden cutting board'},
    'pexels_34644315.jpg': {'board_correct': 1, 'food_correct': 0, 'seen': 'doughnuts arranged on a wooden board'},
    'pexels_35156660.jpg': {'board_correct': 0, 'food_correct': 0, 'seen': 'hands cutting dough on a bare table'},
    'pexels_35657826.jpg': {'board_correct': 0, 'food_correct': 0, 'seen': 'brick wall and wooden beams'},
    'pexels_36430075.jpg': {'board_correct': 0, 'food_correct': 1, 'seen': 'chef cooking in a kitchen'},
}


def main():
    data = json.loads((PROBE / 'summary.json').read_text())
    flagged = [record for record in data['records'] if record['boards'] or record['foods']]
    sample = flagged[:24]
    rows, totals = [], {'board': [0, 0], 'food': [0, 0]}
    for record in sample:
        verdict = VERDICTS.get(record['name'])
        if verdict is None:
            continue
        boards = len(record['boards'])
        foods = len(record['foods'])
        rows.append({'image': record['name'], 'group': record['group'],
                     'board_boxes': boards, 'board_correct': verdict['board_correct'],
                     'food_boxes': foods, 'food_correct': verdict['food_correct'],
                     'seen': verdict['seen']})
        totals['board'][0] += boards
        totals['board'][1] += verdict['board_correct']
        totals['food'][0] += foods
        totals['food'][1] += verdict['food_correct']
    payload = {
        'probe': str(PROBE),
        'source': 'Pexels kitchen photos (free licence), unseen by every model',
        'confidence': 0.5, 'iou': 0.5,
        'images_total': data['total']['images'],
        'images_with_board': data['total']['images_with_board'],
        'images_with_food': data['total']['images_with_food'],
        'flagged_images': len(flagged),
        'verified_images': len(rows),
        'precision_among_detections': {
            'board': {'boxes': totals['board'][0], 'correct': totals['board'][1],
                      'precision': round(totals['board'][1] / totals['board'][0], 3) if totals['board'][0] else None},
            'food': {'boxes': totals['food'][0], 'correct': totals['food'][1],
                     'precision': round(totals['food'][1] / totals['food'][0], 3) if totals['food'][0] else None},
        },
        'failure_mode': 'On staged food photography the board model boxes any tabletop that holds food; '
                        'the three correct boards were wooden boards with dough, bananas or doughnuts on them.',
        'limitations': [
            'No ground-truth boxes: this is precision among detections, recall is unknown.',
            'Only 24 of 111 flagged images were checked; single-rater assistant screening.',
            'Pexels kitchen photos are home/staged scenes, not canteens; not a field false-alarm rate.',
        ],
        'rows': rows,
    }
    (PROBE / 'verification.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(json.dumps({k: payload[k] for k in ('images_total', 'verified_images', 'precision_among_detections')},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
