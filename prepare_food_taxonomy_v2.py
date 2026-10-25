"""Build food_taxonomy_v2: v1 plus empty-label background boards in the training split.

Why: v1 forced every food-like thing into plant_food or animal_food, so bread and bare
wooden boards got called animal_food at 0.64-0.86. Background images teach "board with
nothing on it = no object". Only the training split changes; validation and test stay
byte-identical so v1 numbers remain comparable, and the 22 held-out board-only probe
images are NOT used for training.
"""
import json
import os
from pathlib import Path

BASE = Path(__file__).resolve().parent
V1 = BASE / 'food_taxonomy_v1'
V2 = BASE / 'food_taxonomy_v2'
BOARD = BASE / 'board_dataset'


def link(src, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    try:
        os.link(src, dest)
    except OSError:
        dest.write_bytes(src.read_bytes())


def main():
    manifest = json.loads((V1 / 'manifest.json').read_text())
    background = json.loads((V1 / 'negatives_board.json').read_text())
    probe_ids = {rec['image_id'] for rec in background['images']}

    # Training-split board images with no food annotation at all, from the v1 scan record.
    import importlib.util
    spec = importlib.util.spec_from_file_location('prep', BASE / 'prepare_food_taxonomy.py')
    prep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prep)
    _, food_images, board_images, _, _ = prep.scan('train')
    candidates = sorted(board_images - food_images - probe_ids)
    missing = [i for i in candidates if not (BOARD / 'images' / 'train' / f'{i}.jpg').exists()]
    candidates = [i for i in candidates if i not in missing]

    for split in ['train', 'validation', 'test']:
        for path in (V1 / 'images' / split).glob('*.jpg'):
            link(path, V2 / 'images' / split / path.name)
        for path in (V1 / 'labels' / split).glob('*.txt'):
            link(path, V2 / 'labels' / split / path.name)
    link(V1 / 'attribution.csv', V2 / 'attribution.csv')
    link(V1 / 'negatives_board.json', V2 / 'negatives_board.json')
    if (V1 / 'negatives_board').exists():
        for path in (V1 / 'negatives_board').glob('*.jpg'):
            link(path, V2 / 'negatives_board' / path.name)

    for image_id in candidates:
        link(BOARD / 'images' / 'train' / f'{image_id}.jpg', V2 / 'images' / 'train' / f'{image_id}.jpg')
        (V2 / 'labels' / 'train' / f'{image_id}.txt').write_text('')

    lists = {}
    for split in ['train', 'validation', 'test']:
        images = sorted((V2 / 'images' / split).glob('*.jpg'))
        (V2 / f'{split}.txt').write_text('\n'.join(p.as_posix() for p in images) + '\n')
        lists[split] = len(images)
    (V2 / 'data.yaml').write_text(
        f"path: {V2.as_posix()}\ntrain: train.txt\nval: validation.txt\ntest: test.txt\nnames:\n"
        "  0: plant_food\n  1: animal_food\n")

    labelled = {p.stem for p in (V2 / 'images' / 'train').glob('*.jpg')} - set(candidates)
    overlap = labelled & set(candidates)
    if overlap:
        raise RuntimeError(f'background image also labelled: {sorted(overlap)[:3]}')
    (V2 / 'background_negatives.json').write_text(json.dumps({
        'purpose': 'Training-split cutting-board images with no food annotation, used as empty-label '
                   'backgrounds so bare boards and loaves do not get forced into a food class.',
        'count': len(candidates),
        'image_ids': candidates,
        'excluded_missing_local_files': missing,
        'held_out_probe_not_used_for_training': sorted(probe_ids),
        'source': 'Open Images train split, Cutting board annotations',
    }, indent=2))
    (V2 / 'manifest.json').write_text(json.dumps({
        'parent': str(V1),
        'classes': manifest['classes'],
        'change': f'Added {len(candidates)} empty-label background boards to the training split only.',
        'splits': lists,
        'labelled_train_images': lists['train'] - len(candidates),
        'background_train_images': len(candidates),
        'validation_test_identical_to_v1': True,
        'probe_held_out': True,
    }, indent=2, ensure_ascii=False))
    print('V2_READY', json.dumps({'splits': lists, 'background_added': len(candidates),
                                  'missing_local': len(missing)}))


if __name__ == '__main__':
    main()
