"""Two-class food taxonomy dataset from Open Images: plant_food vs animal_food.

Why two classes: the misuse prototype only knew one coarse "vegetable" class, and it
wrongly fired on cheese. This dataset adds an explicit animal-derived food class so the
rule engine can compare a board's declared food type against a detected one, instead of
guessing. Raw-vs-cooked (生熟) is NOT modelled - Open Images has no annotation for it.

Included classes (Open Images v7 mids):
  plant_food : Vegetable + Broccoli/Cabbage/Carrot/Cucumber/Radish/Tomato subclasses
  animal_food: Seafood, Shellfish, Shrimp, Egg, Cheese
Deliberately excluded and why:
  Chicken, Fish  -> in Open Images these are overwhelmingly live animals, not ingredients
  Salad, Sushi, Hamburger, Bread, Doughnut, Fast food, Food -> prepared/mixed dishes,
  their boxes overlap raw-ingredient boxes and they do not separate 生熟 either.

Images are kept only if every annotation in the buckets above is usable. If a plant box
and an animal box describe the same object (cross-bucket IoU > 0.6) the image is dropped
as ambiguous. Same-bucket duplicates (IoU > 0.5) keep the larger box.
"""
import csv
import hashlib
import json
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image, ImageDraw

BASE = Path(__file__).resolve().parent
RAW = BASE.parent / 'work' / 'datasets'
OUT = BASE / 'food_taxonomy_v1'
CLASSES = ['plant_food', 'animal_food']

BUCKET = {}
for mid in ['/m/0f4s2w', '/m/0hkxq', '/m/0fbw6', '/m/0fj52s', '/m/015x4r', '/m/015x5n', '/m/07j87']:
    BUCKET[mid] = 'plant_food'
for mid in ['/m/06nwz', '/m/0fbdv', '/m/0ll1f78', '/m/033cnk', '/m/01nkt']:
    BUCKET[mid] = 'animal_food'
NAMES = {'/m/0f4s2w': 'Vegetable', '/m/0hkxq': 'Broccoli', '/m/0fbw6': 'Cabbage',
         '/m/0fj52s': 'Carrot', '/m/015x4r': 'Cucumber', '/m/015x5n': 'Radish', '/m/07j87': 'Tomato',
         '/m/06nwz': 'Seafood', '/m/0fbdv': 'Shellfish', '/m/0ll1f78': 'Shrimp',
         '/m/033cnk': 'Egg', '/m/01nkt': 'Cheese'}
# Used only to decide whether a cutting-board image counts as "no labelled food".
FOOD_ALL = set(BUCKET) | {'/m/0grw1', '/m/07030', '/m/0cdn1', '/m/09728', '/m/0jy4k',
                          '/m/01_bhs', '/m/02wbm', '/m/0ch_cf', '/m/09b5t'}
BOARD = '/m/02pdsw'

CAPS = {'train': {'plant_food': 420, 'animal_food': 420},
        'validation': {'plant_food': 70, 'animal_food': 70},
        'test': {'plant_food': 110, 'animal_food': 110}}


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def scan(split):
    """Return {image_id: [(bucket, box)]}, board-only ids, and raw bucket tallies."""
    path = RAW / f'oi_{split}.csv'
    boxes = defaultdict(list)
    food_images = set()
    board_images = set()
    tally = defaultdict(set)
    dropped = defaultdict(int)
    with path.open(encoding='utf-8', newline='') as f:
        reader = csv.reader(f)
        idx = {name: i for i, name in enumerate(next(reader))}
        for row in reader:
            mid = row[idx['LabelName']]
            if mid not in BUCKET and mid != BOARD and mid not in FOOD_ALL:
                continue
            image_id = row[idx['ImageID']]
            if mid in FOOD_ALL:
                if row[idx['IsGroupOf']] == '0' and row[idx['IsDepiction']] == '0':
                    food_images.add(image_id)
            if mid == BOARD:
                board_images.add(image_id)
                continue
            if mid not in BUCKET:
                continue
            if row[idx['IsGroupOf']] != '0' or row[idx['IsDepiction']] != '0':
                dropped['group_of_or_depiction'] += 1
                continue
            try:
                x1, x2, y1, y2 = (float(row[idx[k]]) for k in ('XMin', 'XMax', 'YMin', 'YMax'))
            except ValueError:
                dropped['unparsable_box'] += 1
                continue
            if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
                dropped['invalid_box'] += 1
                continue
            if x2 - x1 < 0.004 or y2 - y1 < 0.004:
                dropped['tiny_box'] += 1
                continue
            boxes[image_id].append((BUCKET[mid], (x1, y1, x2, y2)))
            tally[BUCKET[mid]].add(image_id)
    return boxes, food_images, board_images, tally, dropped


def clean(boxes):
    """Drop ambiguous cross-bucket duplicates, dedupe same-bucket overlaps."""
    kept = []
    for bucket, box in boxes:
        conflict = any(b2 != bucket and iou(box, box2) > 0.6 for b2, box2 in boxes)
        if conflict:
            return None
        kept.append((bucket, box))
    result = []
    for bucket, box in kept:
        dup = False
        for i, (b2, box2) in enumerate(result):
            if b2 != bucket or iou(box, box2) <= 0.5:
                continue
            area, area2 = (box[2] - box[0]) * (box[3] - box[1]), (box2[2] - box2[0]) * (box2[3] - box2[1])
            if area > area2:
                result[i] = (bucket, box)
            dup = True
            break
        if not dup:
            result.append((bucket, box))
    return result


def download(item):
    split, image_id, boxes = item
    dest = OUT / 'images' / split / f'{image_id}.jpg'
    url = f'https://open-images-dataset.s3.amazonaws.com/{split}/{image_id}.jpg'
    for attempt in range(3):
        try:
            if not dest.exists() or dest.stat().st_size == 0:
                r = requests.get(url, timeout=60)
                r.raise_for_status()
                dest.write_bytes(r.content)
            with Image.open(dest) as im:
                im.verify()
            lines = []
            for bucket, (x1, y1, x2, y2) in boxes:
                cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
                lines.append(f'{CLASSES.index(bucket)} {cx:.8f} {cy:.8f} {w:.8f} {h:.8f}')
            (OUT / 'labels' / split / f'{image_id}.txt').write_text('\n'.join(lines) + '\n')
            return {'split': split, 'image_id': image_id, 'url': url,
                    'boxes': {c: sum(b == c for b, _ in boxes) for c in CLASSES},
                    'sha256': hashlib.sha256(dest.read_bytes()).hexdigest()}
        except Exception as exc:  # noqa: BLE001 - retried, raised on last attempt
            if attempt == 2:
                raise RuntimeError(f'{split}/{image_id}: {exc}') from exc


def main():
    random.seed(42)
    stats = {}
    selected = {}
    per_split = {}
    scans = {}
    for split in ['train', 'validation', 'test']:
        boxes, food_images, board_images, tally, dropped = scan(split)
        scans[split] = (food_images, board_images)
        cleaned = {}
        ambiguous = 0
        for image_id, box_list in boxes.items():
            cleaned_boxes = clean(box_list)
            if cleaned_boxes is None:
                ambiguous += 1
                continue
            cleaned[image_id] = cleaned_boxes
        pools = {c: sorted(i for i, b in cleaned.items() if any(x == c for x, _ in b)) for c in CLASSES}
        chosen = set()
        for cls, cap in CAPS[split].items():
            ids = pools[cls]
            random.Random(f'{split}-{cls}-42').shuffle(ids)
            chosen.update(ids[:cap])
        per_split[split] = sorted(chosen)
        selected[split] = {i: cleaned[i] for i in per_split[split]}
        stats[split] = {'available_images_by_class': {c: len(pools[c]) for c in CLASSES},
                        'available_annotated_images_total': len(cleaned),
                        'dropped_ambiguous_cross_class_duplicates': ambiguous,
                        'dropped_annotations': dict(dropped),
                        'selected_images': len(chosen),
                        'selected_boxes': {c: sum(1 for i in chosen for b, _ in cleaned[i] if b == c) for c in CLASSES}}
        print(split, json.dumps(stats[split]), flush=True)

    items = [(s, i, selected[s][i]) for s in selected for i in per_split[s]]
    for split in selected:
        for kind in ['images', 'labels']:
            (OUT / kind / split).mkdir(parents=True, exist_ok=True)
    manifest = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(download, item) for item in items]
        for n, fut in enumerate(as_completed(futures), 1):
            manifest.append(fut.result())
            if n % 100 == 0 or n == len(items):
                print('downloaded', n, '/', len(items), flush=True)

    kept, excluded, seen = [], [], set()
    for rec in sorted(manifest, key=lambda r: (['test', 'validation', 'train'].index(r['split']), r['image_id'])):
        if rec['sha256'] in seen:
            excluded.append(rec)
            continue
        seen.add(rec['sha256'])
        kept.append(rec)
    for split in ['train', 'validation', 'test']:
        lines = [(OUT / 'images' / split / f"{r['image_id']}.jpg").as_posix() for r in kept if r['split'] == split]
        (OUT / f'{split}.txt').write_text('\n'.join(lines) + ('\n' if lines else ''))

    (OUT / 'data.yaml').write_text(
        f"path: {OUT.as_posix()}\ntrain: train.txt\nval: validation.txt\ntest: test.txt\nnames:\n"
        + ''.join(f'  {i}: {c}\n' for i, c in enumerate(CLASSES)))

    # Weak-label negative probe: cutting-board images whose Open Images annotations contain no food at all.
    neg_manifest = build_negatives(scans)
    (OUT / 'negatives_board.json').write_text(json.dumps(neg_manifest, indent=2))

    (OUT / 'manifest.json').write_text(json.dumps({
        'source': 'Open Images v7 official boxes (train v6 file, validation/test v5 files)',
        'classes': CLASSES, 'class_mids': {c: sorted(m for m, b in BUCKET.items() if b == c) for c in CLASSES},
        'class_names': {m: n for m, n in NAMES.items()},
        'deliberately_excluded': ['Chicken (live animal)', 'Fish (mostly live)', 'Salad/Sushi/Hamburger/Bread/Doughnut/Fast food/Food (prepared or mixed, ambiguous 生熟)'],
        'sampling_seed': 42, 'caps': CAPS, 'stats': stats,
        'images': kept, 'excluded_exact_duplicates': excluded,
        'limitations': ['No raw-vs-cooked annotation exists in Open Images.',
                        'Board-only negative images are weak labels: unannotated food may still be present.',
                        'Boxes come from Open Images crowd annotators, not kitchen safety experts.']}, indent=2))

    selected_ids = {r['image_id'] for r in kept}
    attribution = []
    for split in ['train', 'validation', 'test']:
        with (RAW / f'oi_meta_{split}.csv').open(encoding='utf-8', newline='') as f:
            for row in csv.DictReader(f):
                if row['ImageID'] in selected_ids:
                    row['DatasetSplit'] = split
                    attribution.append(row)
    if {r['ImageID'] for r in attribution} != selected_ids:
        raise RuntimeError('Missing attribution rows for some downloaded images')
    with (OUT / 'attribution.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(attribution[0]))
        w.writeheader()
        w.writerows(attribution)

    preview = Image.new('RGB', (960, 720), 'white')
    colors = {0: 'lime', 1: 'red'}
    for i, rec in enumerate([r for r in kept if r['split'] == 'train'][:12]):
        with Image.open(OUT / 'images' / 'train' / f"{rec['image_id']}.jpg") as im:
            image = im.convert('RGB')
        draw = ImageDraw.Draw(image)
        iw, ih = image.size
        for line in (OUT / 'labels' / 'train' / f"{rec['image_id']}.txt").read_text().splitlines():
            c, x, y, w, h = line.split()
            c = int(c)
            draw.rectangle(((float(x) - float(w) / 2) * iw, (float(y) - float(h) / 2) * ih,
                            (float(x) + float(w) / 2) * iw, (float(y) + float(h) / 2) * ih),
                           outline=colors[c], width=3)
        image.thumbnail((320, 160))
        px, py = (i % 3) * 320, (i // 3) * 180
        preview.paste(image, (px, py))
    preview.save(BASE / 'food_taxonomy_preview.jpg')
    print('READY', json.dumps({'per_split': {s: len(v) for s, v in selected.items()},
                               'negative_probe_images': len(neg_manifest['images'])}, ensure_ascii=False), flush=True)


def build_negatives(scans):
    selected_ids = {p.stem for p in (OUT / 'images').rglob('*.jpg')}
    records = []
    dest_dir = OUT / 'negatives_board'
    dest_dir.mkdir(parents=True, exist_ok=True)
    for split in ['validation', 'test']:
        food_images, board_images = scans[split]
        for image_id in sorted(board_images - food_images - selected_ids):
            src = BASE / 'board_dataset' / 'images' / split / f'{image_id}.jpg'
            dest = dest_dir / f'{image_id}.jpg'
            if src.exists() and not dest.exists():
                dest.write_bytes(src.read_bytes())
            records.append({'split': split, 'image_id': image_id,
                            'source': src.as_posix() if src.exists() else None,
                            'exists': dest.exists()})
    return {'purpose': 'Cutting-board images whose Open Images labels contain no food at all. '
                       'Used only as a weak-label false-positive probe, never for training.',
            'images': records}


if __name__ == '__main__':
    main()
