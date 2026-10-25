"""Two-class version of the FoodSeg dataset: merge processed_meat into meat.

Why: processed_meat (sausage, fried meat) was the weakest class by far (AP50 0.086 on 19 test
images) and it competes with meat for the same visual features. Merging keeps every image,
box and split byte-identical - only the class index changes - so the comparison with
foodseg_v2 is clean.

New classes: 0 = meat (steak/pork/chicken-duck/lamb/sausage/fried meat), 1 = vegetable.
"""
import json
import os
from pathlib import Path

BASE = Path(__file__).resolve().parent
SRC = BASE / 'foodseg_dataset_v2'
OUT = BASE / 'foodseg_two_class'
CLASSES = ['meat', 'vegetable']
REMAP = {0: 0, 1: 0, 2: 1}


def link(src, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    try:
        os.link(src, dest)
    except OSError:
        dest.write_bytes(src.read_bytes())


def main():
    manifest = json.loads((SRC / 'manifest.json').read_text())
    for split in ['train', 'validation', 'test']:
        for path in (SRC / 'images' / split).glob('*.jpg'):
            link(path, OUT / 'images' / split / path.name)
        for path in (SRC / 'labels' / split).glob('*.txt'):
            lines = []
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                cls, *rest = line.split()
                lines.append(' '.join([str(REMAP[int(cls)]), *rest]))
            (OUT / 'labels' / split / path.name).parent.mkdir(parents=True, exist_ok=True)
            (OUT / 'labels' / split / path.name).write_text('\n'.join(lines) + ('\n' if lines else ''))
    for name in ['negatives_board.json']:
        if (SRC / name).exists():
            link(SRC / name, OUT / name)
    if (SRC / 'negatives_board').exists():
        for path in (SRC / 'negatives_board').glob('*.jpg'):
            link(path, OUT / 'negatives_board' / path.name)

    lists, boxes = {}, {}
    for split in ['train', 'validation', 'test']:
        paths = sorted((OUT / 'images' / split).glob('*.jpg'))
        (OUT / f'{split}.txt').write_text('\n'.join(p.as_posix() for p in paths) + '\n')
        lists[split] = len(paths)
        counts = {c: 0 for c in CLASSES}
        for label in (OUT / 'labels' / split).glob('*.txt'):
            for line in label.read_text().splitlines():
                if line.strip():
                    counts[CLASSES[int(line.split()[0])]] += 1
        boxes[split] = counts
    (OUT / 'data.yaml').write_text(
        f"path: {OUT.as_posix()}\ntrain: train.txt\nval: validation.txt\ntest: test.txt\nnames:\n"
        + ''.join(f'  {i}: {c}\n' for i, c in enumerate(CLASSES)))
    (OUT / 'manifest.json').write_text(json.dumps({
        'parent': str(SRC), 'classes': CLASSES, 'remap': {'0': 0, '1': 0, '2': 1},
        'change': 'processed_meat merged into meat; images and splits unchanged',
        'images': lists, 'boxes': boxes,
        'reason': 'processed_meat AP50 was 0.086 on 19 test images and competed with meat.',
        'caveats': manifest.get('caveats', []) + ['Merged class means sausage/fried meat is no longer separable.'],
    }, indent=2, ensure_ascii=False))
    print('TWO_CLASS_READY', json.dumps({'splits': lists, 'boxes': boxes}, ensure_ascii=False))


if __name__ == '__main__':
    main()
