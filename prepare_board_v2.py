"""Build board_dataset_v2 = Open Images board set + 46 verified real-kitchen images.

Adds two things to the original board dataset's training split:
  - 2 real kitchen photos where the board model's own box was confirmed to be a cutting board
  - 44 real kitchen photos with NO board, as empty-label hard negatives

Those 44 are the exact objects that produced false positives in the kitchen probe (bowls,
plates, cups, mixers, washing machines, cabinets, mats). Validation and test splits stay
byte-identical to board_dataset so the original numbers remain comparable, and the 92 kitchen
images not used here stay available as a hold-out.
"""
import json
import os
from pathlib import Path

from PIL import Image

BASE = Path(__file__).resolve().parent
SRC = BASE / 'board_dataset'
PROBE = BASE / 'real_kitchen_probe'
OUT = BASE / 'board_dataset_v2'
LABELS = json.loads((BASE / 'board_real_labels.json').read_text())
PROBE_DATA = json.loads((PROBE / 'domain_probe.json').read_text())


def link(src, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    try:
        os.link(src, dest)
    except OSError:
        dest.write_bytes(src.read_bytes())


def main():
    for split in ['train', 'validation', 'test']:
        for path in (SRC / 'images' / split).glob('*.jpg'):
            link(path, OUT / 'images' / split / path.name)
        for path in (SRC / 'labels' / split).glob('*.txt'):
            link(path, OUT / 'labels' / split / path.name)
    for name in ['attribution.csv']:
        if (SRC / name).exists():
            link(SRC / name, OUT / name)

    by_id = {record['image'].replace('.jpg', ''): record for record in PROBE_DATA['images']}
    added = {'positives': 0, 'negatives': 0, 'skipped_missing': 0}

    def resolve(prefix):
        for full, record in by_id.items():
            if full.startswith(prefix):
                return full, record
        return None, None

    for image_id in LABELS['no_board']:
        full, _ = resolve(image_id)
        source = PROBE / 'images' / f'{full}.jpg' if full else None
        if source is None or not source.exists():
            added['skipped_missing'] += 1
            continue
        link(source, OUT / 'images' / 'train' / f'{full}.jpg')
        (OUT / 'labels' / 'train' / f'{full}.txt').write_text('')
        added['negatives'] += 1
    for image_id, box_number in LABELS['board_accept_box'].items():
        full, record = resolve(image_id)
        source = PROBE / 'images' / f'{full}.jpg' if full else None
        if source is None or not source.exists() or record is None:
            added['skipped_missing'] += 1
            continue
        box = record['boards'][box_number - 1]['xyxy']
        with Image.open(source) as image:
            width, height = image.size
        x1, y1, x2, y2 = box
        line = (f'0 {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} '
                f'{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}')
        link(source, OUT / 'images' / 'train' / f'{image_id}.jpg')
        (OUT / 'labels' / 'train' / f'{image_id}.txt').write_text(line + '\n')
        added['positives'] += 1

    lists = {}
    for split in ['train', 'validation', 'test']:
        paths = sorted((OUT / 'images' / split).glob('*.jpg'))
        (OUT / f'{split}.txt').write_text('\n'.join(p.as_posix() for p in paths) + '\n')
        lists[split] = len(paths)
    (OUT / 'data.yaml').write_text(
        f"path: {OUT.as_posix()}\ntrain: train.txt\nval: validation.txt\ntest: test.txt\nnames:\n  0: cutting_board\n")
    (OUT / 'manifest.json').write_text(json.dumps({
        'parent': str(SRC),
        'change': 'training split only: +2 verified real-kitchen boards, +44 verified real-kitchen hard negatives',
        'images': lists,
        'added': added,
        'labels_record': 'board_real_labels.json',
        'holdout': 'the other 92 real_kitchen_probe images are NOT in this dataset and are used for evaluation',
        'caveat': 'hard negatives are images where the assistant confirmed no cutting board is visible; they are not exhaustive scene labels.',
    }, indent=2, ensure_ascii=False))
    print('BOARD_V2_READY', json.dumps({'splits': lists, 'added': added}, ensure_ascii=False))


if __name__ == '__main__':
    main()
