"""Build a detection dataset from FoodSeg103 masks: meat / processed_meat / vegetable.

Why: the Open Images food models have no real meat annotation, only plated seafood/egg/cheese.
FoodSeg103 is a Chinese-dish segmentation benchmark whose labels include steak, pork,
chicken duck, lamb, sausage and fried meat, plus a large vegetable list. Masks are converted
to boxes per class with connected components.

Honest caveats written into the manifest:
  - FoodSeg103 labels are dish/ingredient categories, NOT a verified raw-vs-cooked annotation.
    "meat" here means the steak/pork/chicken-duck/lamb categories; many of those pixels are
    cooked dishes. Do not claim 生熟 capability from this.
  - The HF mirror declares apache-2.0; the original benchmark was published by LARC/SMU.
    Verify the licence yourself before any commercial use.
  - Official splits are train/validation only. Validation is halved into val/test with a
    fixed seed, so the test split is ours, not an official benchmark split.

Files are written only after a sample is accepted (cap + duplicate checks), so the image
folders and the split lists always agree.
"""
import hashlib
import io
import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

BASE = Path(__file__).resolve().parent
RAW = BASE.parent / 'work' / 'datasets' / 'foodseg103'
OUT = BASE / 'foodseg_dataset'
CLASSES = ['meat', 'processed_meat', 'vegetable']
CLASS_IDS = {
    'meat': {46, 47, 48, 51},          # steak, pork, chicken duck, lamb
    'processed_meat': {49, 50},        # sausage, fried meat
    'vegetable': set(range(69, 97)),   # eggplant .. French beans
}
ALL_IDS = set().union(*CLASS_IDS.values())
CAPS = {'train': {'meat': 700, 'processed_meat': 350, 'vegetable': 1200},
        'validation': {'meat': 140, 'processed_meat': 70, 'vegetable': 220},
        'test': {'meat': 140, 'processed_meat': 70, 'vegetable': 220}}
MIN_AREA_FRACTION = 0.002
MAX_BOXES_PER_CLASS = 20


def boxes_from_mask(mask, height, width):
    found = []
    for name, ids in CLASS_IDS.items():
        binary = np.isin(mask, list(ids)).astype(np.uint8)
        if not binary.any():
            continue
        count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        candidates = []
        for index in range(1, count):
            x, y, w, h, area = stats[index]
            if area < MIN_AREA_FRACTION * height * width or w < 6 or h < 6:
                continue
            candidates.append((area, x, y, w, h))
        candidates.sort(reverse=True)
        for _, x, y, w, h in candidates[:MAX_BOXES_PER_CLASS]:
            found.append((name, x, y, w, h))
    return found


def rows(path):
    handle = pq.ParquetFile(path)
    for batch in handle.iter_batches(batch_size=64, columns=['image', 'label', 'classes_on_image', 'id']):
        for row in batch.to_pylist():
            if not ALL_IDS.intersection(row['classes_on_image'] or []):
                continue
            yield row


def render(row):
    image = Image.open(io.BytesIO(row['image']['bytes'])).convert('RGB')
    mask_image = Image.open(io.BytesIO(row['label']['bytes']))
    if mask_image.size != image.size:
        mask_image = mask_image.resize(image.size, Image.NEAREST)
    boxes = boxes_from_mask(np.array(mask_image), image.size[1], image.size[0])
    if not boxes:
        return None
    width, height = image.size
    lines = []
    for cls, x, y, w, h in boxes:
        lines.append(f'{CLASSES.index(cls)} {(x + w / 2) / width:.6f} {(y + h / 2) / height:.6f} '
                     f'{w / width:.6f} {h / height:.6f}')
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=92)
    payload = buffer.getvalue()
    return {'id': int(row['id']), 'name': f'fs{row["id"]:06d}', 'bytes': payload,
            'label': '\n'.join(lines) + '\n',
            'boxes': {c: sum(b == c for b, *_ in boxes) for c in CLASSES},
            'sha256': hashlib.sha256(payload).hexdigest()}


def main():
    for split in CAPS:
        for kind in ['images', 'labels']:
            (OUT / kind / split).mkdir(parents=True, exist_ok=True)
        for path in (OUT / 'images' / split).glob('*.jpg'):
            path.unlink()
        for path in (OUT / 'labels' / split).glob('*.txt'):
            path.unlink()

    chosen = {split: [] for split in CAPS}
    counts = {split: {c: 0 for c in CLASSES} for split in CAPS}
    seen_ids, seen_hashes = set(), set()
    skipped_duplicate = 0

    def take(row, split):
        nonlocal skipped_duplicate
        if row['id'] in seen_ids:
            return
        record = render(row)
        if record is None:
            return
        names = [c for c, n in record['boxes'].items() if n]
        if not any(counts[split][c] < CAPS[split][c] for c in names):
            return
        if record['sha256'] in seen_hashes:
            skipped_duplicate += 1
            return
        seen_ids.add(row['id'])
        seen_hashes.add(record['sha256'])
        (OUT / 'images' / split / f'{record["name"]}.jpg').write_bytes(record['bytes'])
        (OUT / 'labels' / split / f'{record["name"]}.txt').write_text(record['label'])
        chosen[split].append({k: record[k] for k in ('id', 'name', 'boxes')})
        for cls in names:
            counts[split][cls] += 1

    for shard in sorted(RAW.glob('train-*.parquet')):
        for row in rows(shard):
            take(row, 'train')
        print('train so far', len(chosen['train']), counts['train'], flush=True)

    validation_rows = [row for row in rows(RAW / 'validation-00000-of-00001.parquet')]
    order = np.random.default_rng(42).permutation(len(validation_rows))
    split_of = {index: ('validation' if rank % 2 == 0 else 'test') for rank, index in enumerate(order)}
    for index, row in enumerate(validation_rows):
        take(row, split_of[index])
    print('val', len(chosen['validation']), counts['validation'], flush=True)
    print('test', len(chosen['test']), counts['test'], flush=True)

    lists = {}
    for split in CAPS:
        paths = sorted((OUT / 'images' / split).glob('*.jpg'))
        (OUT / f'{split}.txt').write_text('\n'.join(p.as_posix() for p in paths) + '\n')
        lists[split] = len(paths)
        if len(paths) != len(chosen[split]):
            raise RuntimeError(f'{split}: {len(paths)} files on disk but {len(chosen[split])} accepted')
    (OUT / 'data.yaml').write_text(
        f"path: {OUT.as_posix()}\ntrain: train.txt\nval: validation.txt\ntest: test.txt\nnames:\n"
        + ''.join(f'  {i}: {c}\n' for i, c in enumerate(CLASSES)))
    (OUT / 'manifest.json').write_text(json.dumps({
        'source': 'FoodSeg103 via hf-mirror (datasets/EduardoPacheco/FoodSeg103)',
        'license': 'HF dataset card declares apache-2.0; original benchmark by LARC/SMU. Verify before commercial use.',
        'classes': CLASSES,
        'class_ids': {c: sorted(v) for c, v in CLASS_IDS.items()},
        'caps': CAPS,
        'images': lists,
        'images_per_class': counts,
        'boxes': {split: {c: sum(r['boxes'][c] for r in chosen[split]) for c in CLASSES} for split in CAPS},
        'mask_to_box': {'method': 'connected components on the class mask',
                        'min_area_fraction': MIN_AREA_FRACTION,
                        'max_boxes_per_class': MAX_BOXES_PER_CLASS},
        'splits': 'official train/validation; validation halved into validation/test with seed 42',
        'duplicates_skipped_by_image_hash': skipped_duplicate,
        'caveats': ['Dish/ingredient categories, not a verified raw-vs-cooked annotation.',
                    'mask-derived boxes approximate visible ingredient regions.',
                    'test split is ours, not the official benchmark test split.'],
    }, indent=2, ensure_ascii=False))
    print('READY', json.dumps({'splits': lists, 'skipped_duplicates': skipped_duplicate}))


if __name__ == '__main__':
    main()
