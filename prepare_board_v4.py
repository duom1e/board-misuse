"""board_dataset_v4: board_v2 + verified hard negatives from Pexels (tables, islands, plating).

The Pexels probe showed the board model boxes any tabletop holding food (precision 0.167).
This adds the images where that happened and a human confirmed there is no cutting board.
Those images are then EXCLUDED from the evaluation list so the check stays honest.
"""
import json
import os
from pathlib import Path

BASE = Path(__file__).resolve().parent
SRC = BASE / 'board_dataset_v2'
OUT = BASE / 'board_dataset_v4'
PROBE = BASE / 'pexels_kitchen' / 'probe'
VERIFICATION = json.loads((PROBE / 'verification.json').read_text())


def link(src, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    try:
        os.link(src, dest)
    except OSError:
        dest.write_bytes(src.read_bytes())


def main():
    rows = VERIFICATION['rows']
    negatives = [row for row in rows if row['board_boxes'] and row['board_correct'] == 0]
    positives = [row for row in rows if row['board_correct'] > 0]
    by_name = {record['name']: record for record in json.loads((PROBE / 'summary.json').read_text())['records']}

    for split in ['train', 'validation', 'test']:
        for path in (SRC / 'images' / split).glob('*.jpg'):
            link(path, OUT / 'images' / split / path.name)
        for path in (SRC / 'labels' / split).glob('*.txt'):
            link(path, OUT / 'labels' / split / path.name)
    if (SRC / 'attribution.csv').exists():
        link(SRC / 'attribution.csv', OUT / 'attribution.csv')

    added = 0
    for row in negatives:
        record = by_name.get(row['image'])
        if record is None:
            continue
        source = Path(record['image'])
        link(source, OUT / 'images' / 'train' / source.name)
        (OUT / 'labels' / 'train' / f'{source.stem}.txt').write_text('')
        added += 1
    for row in positives:
        record = by_name.get(row['image'])
        if record is None or not record['boards']:
            continue
        source = Path(record['image'])
        with __import__('PIL.Image').Image.open(source) as image:
            width, height = image.size
        box = record['boards'][0]['xyxy']
        line = (f'0 {(box[0] + box[2]) / 2 / width:.6f} {(box[1] + box[3]) / 2 / height:.6f} '
                f'{(box[2] - box[0]) / width:.6f} {(box[3] - box[1]) / height:.6f}')
        link(source, OUT / 'images' / 'train' / source.name)
        (OUT / 'labels' / 'train' / f'{source.stem}.txt').write_text(line + '\n')
        added += 1

    lists = {}
    for split in ['train', 'validation', 'test']:
        paths = sorted((OUT / 'images' / split).glob('*.jpg'))
        (OUT / f'{split}.txt').write_text('\n'.join(p.as_posix() for p in paths) + '\n')
        lists[split] = len(paths)
    (OUT / 'data.yaml').write_text(
        f"path: {OUT.as_posix()}\ntrain: train.txt\nval: validation.txt\ntest: test.txt\nnames:\n  0: cutting_board\n")

    used = {row['image'] for row in rows}
    holdout = sorted(str(path) for path in (BASE / 'pexels_kitchen').rglob('*.jpg')
                     if path.parent.name in ('images', 'images2', 'canteen') and path.name not in used)
    (OUT / 'pexels_holdout.txt').write_text('\n'.join(holdout) + '\n')
    (OUT / 'manifest.json').write_text(json.dumps({
        'parent': str(SRC),
        'change': f'training split only: +{len(negatives)} verified no-board Pexels scenes, '
                  f'+{len(positives)} verified real boards',
        'images': lists,
        'hard_negatives': [row['image'] for row in negatives],
        'positives': [row['image'] for row in positives],
        'pexels_holdout_images': len(holdout),
        'eval_rule': 'the 24 verified probe images are in training, so evaluation uses the remaining Pexels photos',
        'caveat': 'hard negatives are images where a human judged no cutting board is visible; single-rater.',
    }, indent=2, ensure_ascii=False))
    print(json.dumps({'splits': lists, 'negatives': len(negatives), 'positives': len(positives),
                      'holdout': len(holdout)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
