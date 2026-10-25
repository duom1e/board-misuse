"""foodseg_dataset_v2: the FoodSeg103 set plus real board photos with no food labels.

Why: the FoodSeg model still called a wooden workbench "vegetable" at 0.80. Background
images (a board with nothing on it) are the standard fix. Sources are Open Images cutting
board photos whose annotations contain no food at all.

Only the training split changes; validation and test stay byte-identical so the foodseg_v1
numbers remain comparable. The 22 held-out board-only probe images are excluded from
training so they stay usable as a false-positive check.
"""
import json
import os
from pathlib import Path

BASE = Path(__file__).resolve().parent
SRC = BASE / 'foodseg_dataset'
OUT = BASE / 'foodseg_dataset_v2'
BOARD = BASE / 'board_dataset'
CAP = 180


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
    probe = json.loads((SRC / 'negatives_board.json').read_text())
    probe_ids = {rec['image_id'] for rec in probe['images']}

    import importlib.util
    spec = importlib.util.spec_from_file_location('prep', BASE / 'prepare_food_taxonomy.py')
    prep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prep)
    _, food_images, board_images, _, _ = prep.scan('train')
    pool = sorted(board_images - food_images - probe_ids)
    import random
    random.Random('foodseg-v2-backgrounds').shuffle(pool)
    pool = [i for i in pool if (BOARD / 'images' / 'train' / f'{i}.jpg').exists()]
    backgrounds = pool[:CAP]
    if len(backgrounds) < CAP:
        print(f'only {len(backgrounds)} background images available')

    for split in ['train', 'validation', 'test']:
        for path in (SRC / 'images' / split).glob('*.jpg'):
            link(path, OUT / 'images' / split / path.name)
        for path in (SRC / 'labels' / split).glob('*.txt'):
            link(path, OUT / 'labels' / split / path.name)
    for name in ['negatives_board.json', 'attribution.csv']:
        if (SRC / name).exists():
            link(SRC / name, OUT / name)
    if (SRC / 'negatives_board').exists():
        for path in (SRC / 'negatives_board').glob('*.jpg'):
            link(path, OUT / 'negatives_board' / path.name)

    for image_id in backgrounds:
        link(BOARD / 'images' / 'train' / f'{image_id}.jpg', OUT / 'images' / 'train' / f'{image_id}.jpg')
        (OUT / 'labels' / 'train' / f'{image_id}.txt').write_text('')

    lists = {}
    for split in ['train', 'validation', 'test']:
        paths = sorted((OUT / 'images' / split).glob('*.jpg'))
        (OUT / f'{split}.txt').write_text('\n'.join(p.as_posix() for p in paths) + '\n')
        lists[split] = len(paths)
    (OUT / 'data.yaml').write_text(
        f"path: {OUT.as_posix()}\ntrain: train.txt\nval: validation.txt\ntest: test.txt\nnames:\n"
        + ''.join(f'  {i}: {c}\n' for i, c in enumerate(manifest['classes'])))
    (OUT / 'background_negatives.json').write_text(json.dumps({
        'count': len(backgrounds), 'image_ids': backgrounds,
        'source': 'Open Images train split, Cutting board with no food annotation',
        'fraction_of_labelled': round(len(backgrounds) / (lists['train'] - len(backgrounds)), 3),
        'probe_excluded': sorted(probe_ids)}, indent=2))
    (OUT / 'manifest.json').write_text(json.dumps({
        'parent': str(SRC), 'classes': manifest['classes'],
        'change': f'Added {len(backgrounds)} empty-label board photos to the training split only.',
        'images': lists,
        'validation_test_identical_to_parent': True,
        'note': 'Board photos are weak negatives: unannotated food may still be present, but the point is to '
                'teach "bare board = no food class".',
    }, indent=2, ensure_ascii=False))
    print('V2_READY', json.dumps({'splits': lists, 'backgrounds': len(backgrounds)}))


if __name__ == '__main__':
    main()
