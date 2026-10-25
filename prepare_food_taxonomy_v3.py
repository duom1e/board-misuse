"""Build food_taxonomy_v3: three food classes plus background boards.

Change from v1/v2: a third class `other_food` (Bread, Sushi, Hamburger, Doughnut) so that
"edible, but neither vegetable nor animal protein" has its own label instead of being
forced into plant_food or animal_food. That forcing is why v1 called bread animal_food
and v2 called it plant_food.

Deliberate exclusions, unchanged reasoning:
  Salad      -> boxes overlap Vegetable almost everywhere; a third class would just relabel the same object
  Fast food  -> superset of Hamburger/Sushi boxes, would create duplicate labels
  Chicken/Fish -> live animals in Open Images, not ingredients

Backgrounds: 120 training-split cutting-board images with no food annotation, kept at about
11% of the labelled set (v2 used 156/838 = 19% and lost too much recall).

The 22 held-out board-only probe images stay out of training, but they were already used to
choose between v1 and v2, so they are NOT pristine. This dataset's own test split is new and
is the primary comparison for v3.
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
V1 = BASE / 'food_taxonomy_v1'
OUT = BASE / 'food_taxonomy_v3'
BOARD_DIR = BASE / 'board_dataset'
CLASSES = ['plant_food', 'animal_food', 'other_food']

BUCKET = {}
for mid in ['/m/0f4s2w', '/m/0hkxq', '/m/0fbw6', '/m/0fj52s', '/m/015x4r', '/m/015x5n', '/m/07j87']:
    BUCKET[mid] = 'plant_food'
for mid in ['/m/06nwz', '/m/0fbdv', '/m/0ll1f78', '/m/033cnk', '/m/01nkt']:
    BUCKET[mid] = 'animal_food'
for mid in ['/m/09728', '/m/07030', '/m/0cdn1', '/m/0jy4k']:
    BUCKET[mid] = 'other_food'
NAMES = {'/m/0f4s2w': 'Vegetable', '/m/0hkxq': 'Broccoli', '/m/0fbw6': 'Cabbage', '/m/0fj52s': 'Carrot',
         '/m/015x4r': 'Cucumber', '/m/015x5n': 'Radish', '/m/07j87': 'Tomato', '/m/06nwz': 'Seafood',
         '/m/0fbdv': 'Shellfish', '/m/0ll1f78': 'Shrimp', '/m/033cnk': 'Egg', '/m/01nkt': 'Cheese',
         '/m/09728': 'Bread', '/m/07030': 'Sushi', '/m/0cdn1': 'Hamburger', '/m/0jy4k': 'Doughnut'}
FOOD_ALL = set(BUCKET) | {'/m/0grw1', '/m/01_bhs', '/m/02wbm', '/m/0ch_cf', '/m/09b5t'}
BOARD = '/m/02pdsw'
BACKGROUND_CAP = 120
CAPS = {'train': {'plant_food': 420, 'animal_food': 420, 'other_food': 300},
        'validation': {'plant_food': 70, 'animal_food': 70, 'other_food': 50},
        'test': {'plant_food': 110, 'animal_food': 110, 'other_food': 90}}


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
    boxes = defaultdict(list)
    food_images = set()
    board_images = set()
    dropped = defaultdict(int)
    with (RAW / f'oi_{split}.csv').open(encoding='utf-8', newline='') as f:
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
    return boxes, food_images, board_images, dropped


def clean(boxes):
    for bucket, box in boxes:
        if any(other != bucket and iou(box, other_box) > 0.6 for other, other_box in boxes):
            return None
    result = []
    for bucket, box in boxes:
        duplicate = False
        for index, (other, other_box) in enumerate(result):
            if other != bucket or iou(box, other_box) <= 0.5:
                continue
            area = (box[2] - box[0]) * (box[3] - box[1])
            other_area = (other_box[2] - other_box[0]) * (other_box[3] - other_box[1])
            if area > other_area:
                result[index] = (bucket, box)
            duplicate = True
            break
        if not duplicate:
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
        except Exception as exc:  # noqa: BLE001
            if attempt == 2:
                raise RuntimeError(f'{split}/{image_id}: {exc}') from exc


def main():
    random.seed(42)
    # Reuse already-downloaded v1 images as hard links so only new ids hit the network.
    reused = 0
    if (V1 / 'manifest.json').exists():
        for rec in json.loads((V1 / 'manifest.json').read_text())['images']:
            src = V1 / 'images' / rec['split'] / f"{rec['image_id']}.jpg"
            dest = OUT / 'images' / rec['split'] / f"{rec['image_id']}.jpg"
            if src.exists() and not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                try:
                    import os
                    os.link(src, dest)
                except OSError:
                    dest.write_bytes(src.read_bytes())
                reused += 1

    stats, selected, per_split, scans = {}, {}, {}, {}
    for split in ['train', 'validation', 'test']:
        boxes, food_images, board_images, dropped = scan(split)
        scans[split] = (food_images, board_images)
        cleaned, ambiguous = {}, 0
        for image_id, box_list in boxes.items():
            result = clean(box_list)
            if result is None:
                ambiguous += 1
                continue
            cleaned[image_id] = result
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

    # Background boards: training split only, empty labels.
    probe_ids = set()
    if (V1 / 'negatives_board.json').exists():
        probe_ids = {rec['image_id'] for rec in json.loads((V1 / 'negatives_board.json').read_text())['images']}
    food_images, board_images = scans['train']
    pool = sorted(board_images - food_images - probe_ids - set(per_split['train']))
    random.Random('backgrounds-42').shuffle(pool)
    backgrounds = [i for i in pool if (BOARD_DIR / 'images' / 'train' / f'{i}.jpg').exists()][:BACKGROUND_CAP]

    items = [(s, i, selected[s][i]) for s in selected for i in per_split[s]]
    for split in selected:
        for kind in ['images', 'labels']:
            (OUT / kind / split).mkdir(parents=True, exist_ok=True)
    # The reuse step above linked v1 images by their v1 split, which can leave files that
    # v3 did not select for that split. Remove any image not selected here.
    keep = {i for i in per_split['train']} | set(backgrounds)
    keep_by_split = {'train': keep, 'validation': set(per_split['validation']), 'test': set(per_split['test'])}
    pruned = 0
    for split, ids in keep_by_split.items():
        for path in (OUT / 'images' / split).glob('*.jpg'):
            if path.stem not in ids:
                path.unlink()
                (OUT / 'labels' / split / f'{path.stem}.txt').unlink(missing_ok=True)
                pruned += 1
    # Re-link any selected image that already exists in v1 under a different split folder.
    v1_images = {}
    if (V1 / 'manifest.json').exists():
        for rec in json.loads((V1 / 'manifest.json').read_text())['images']:
            v1_images[rec['image_id']] = V1 / 'images' / rec['split'] / f"{rec['image_id']}.jpg"
    relinked = 0
    for split, image_id, _ in items:
        dest = OUT / 'images' / split / f'{image_id}.jpg'
        src = v1_images.get(image_id)
        if dest.exists() or src is None or not src.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            import os
            os.link(src, dest)
        except OSError:
            dest.write_bytes(src.read_bytes())
        relinked += 1
    manifest = []
    pending = [it for it in items if not (OUT / 'images' / it[0] / f'{it[1]}.jpg').exists()]
    if pending:
        with ThreadPoolExecutor(max_workers=8) as pool_exec:
            futures = [pool_exec.submit(download, item) for item in pending]
            for n, fut in enumerate(as_completed(futures), 1):
                manifest.append(fut.result())
                if n % 100 == 0 or n == len(pending):
                    print('downloaded', n, '/', len(pending), flush=True)
    written = []
    for split, image_id, boxes in items:
        path = OUT / 'images' / split / f'{image_id}.jpg'
        lines = [f'{CLASSES.index(bucket)} {(box[0] + box[2]) / 2:.8f} {(box[1] + box[3]) / 2:.8f} '
                 f'{box[2] - box[0]:.8f} {box[3] - box[1]:.8f}' for bucket, box in boxes]
        (OUT / 'labels' / split / f'{image_id}.txt').write_text('\n'.join(lines) + '\n')
        written.append({'split': split, 'image_id': image_id, 'url': f'https://open-images-dataset.s3.amazonaws.com/{split}/{image_id}.jpg',
                        'boxes': {c: sum(b == c for b, _ in boxes) for c in CLASSES},
                        'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})

    for image_id in backgrounds:
        src = BOARD_DIR / 'images' / 'train' / f'{image_id}.jpg'
        dest = OUT / 'images' / 'train' / f'{image_id}.jpg'
        if not dest.exists():
            try:
                import os
                os.link(src, dest)
            except OSError:
                dest.write_bytes(src.read_bytes())
        (OUT / 'labels' / 'train' / f'{image_id}.txt').write_text('')

    lists = {}
    for split in ['train', 'validation', 'test']:
        paths = sorted((OUT / 'images' / split).glob('*.jpg'))
        (OUT / f'{split}.txt').write_text('\n'.join(p.as_posix() for p in paths) + '\n')
        lists[split] = len(paths)
    (OUT / 'data.yaml').write_text(f"path: {OUT.as_posix()}\ntrain: train.txt\nval: validation.txt\ntest: test.txt\nnames:\n"
                                   + ''.join(f'  {i}: {c}\n' for i, c in enumerate(CLASSES)))
    (OUT / 'background_negatives.json').write_text(json.dumps({
        'count': len(backgrounds), 'image_ids': backgrounds,
        'fraction_of_labelled': round(len(backgrounds) / max(1, lists['train'] - len(backgrounds)), 3),
        'reason': 'v2 used 156/838 (0.19) and lost recall; this keeps about 11 percent.',
        'probe_not_used': sorted(probe_ids)}, indent=2))
    if (V1 / 'attribution.csv').exists():
        raw = (V1 / 'attribution.csv').read_text(encoding='utf-8-sig')
    attribution_rows = []
    selected_ids = {r['image_id'] for r in written}
    for split in ['train', 'validation', 'test']:
        with (RAW / f'oi_meta_{split}.csv').open(encoding='utf-8', newline='') as f:
            for row in csv.DictReader(f):
                if row['ImageID'] in selected_ids:
                    row['DatasetSplit'] = split
                    attribution_rows.append(row)
    missing = selected_ids - {r['ImageID'] for r in attribution_rows}
    if missing:
        raise RuntimeError(f'Missing attribution for {len(missing)} images')
    with (OUT / 'attribution.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(attribution_rows[0]))
        writer.writeheader()
        writer.writerows(attribution_rows)
    (OUT / 'negatives_board.json').write_text((V1 / 'negatives_board.json').read_text(encoding='utf-8'))
    (OUT / 'manifest.json').write_text(json.dumps({
        'classes': CLASSES,
        'class_mids': {c: sorted(m for m, b in BUCKET.items() if b == c) for c in CLASSES},
        'class_names': NAMES,
        'excluded': ['Salad (overlaps Vegetable)', 'Fast food (superset of Hamburger/Sushi)',
                     'Chicken/Fish (live animals)'],
        'caps': CAPS, 'backgrounds': len(backgrounds), 'reused_v1_images': reused,
        'stats': stats, 'splits': lists,
        'note': 'Test split differs from v1/v2 (third class added); compare within v3, not across versions.',
    }, indent=2, ensure_ascii=False))

    preview = Image.new('RGB', (960, 720), 'white')
    colors = {0: 'lime', 1: 'red', 2: 'blue'}
    train_images = sorted((OUT / 'images' / 'train').glob('*.jpg'))
    shown = 0
    for path in train_images:
        label = OUT / 'labels' / 'train' / f'{path.stem}.txt'
        text = label.read_text().strip()
        if not text:
            continue
        with Image.open(path) as im:
            image = im.convert('RGB')
        draw = ImageDraw.Draw(image)
        iw, ih = image.size
        for line in text.splitlines():
            c, x, y, w, h = line.split()
            draw.rectangle(((float(x) - float(w) / 2) * iw, (float(y) - float(h) / 2) * ih,
                            (float(x) + float(w) / 2) * iw, (float(y) + float(h) / 2) * ih),
                           outline=colors[int(c)], width=3)
        image.thumbnail((320, 160))
        preview.paste(image, ((shown % 3) * 320, (shown // 3) * 180))
        shown += 1
        if shown >= 12:
            break
    preview.save(BASE / 'food_taxonomy_v3_preview.jpg')
    print('V3_READY', json.dumps({'splits': lists, 'backgrounds': len(backgrounds), 'reused_v1': reused,
                                  'new_downloads': len(manifest)}))


if __name__ == '__main__':
    main()
