"""board_dataset_v3: original board set + both labelling batches, and a fresh kitchen eval set.

Labels now cover 88 real kitchen photos (10 verified boards, 78 verified no-board scenes).
Those images are all used for training, so the old hold-out is no longer held out; this script
also downloads a NEW probe of kitchen photos that no model has seen.
"""
import csv
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image

BASE = Path(__file__).resolve().parent
SRC = BASE / 'board_dataset'
PROBE_A = BASE / 'real_kitchen_probe'
PROBE_B = BASE / 'real_kitchen_probe2'
OUT = BASE / 'board_dataset_v3'
RAW = BASE.parent / 'work' / 'datasets'
MIDS = {'/m/058qzx': 'Kitchen knife', '/m/03_wxk': 'Kitchenware', '/m/03hlz0c': 'Kitchen utensil',
        '/m/0h99cwc': 'Kitchen appliance', '/m/02pdsw': 'Cutting board'}
NEW_PROBE = 90


def link(src, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    try:
        os.link(src, dest)
    except OSError:
        dest.write_bytes(src.read_bytes())


def used_image_ids():
    ids = set()
    for name in ['board_dataset', 'board_dataset_v2', 'food_taxonomy_v1', 'food_taxonomy_v2',
                 'food_taxonomy_v3', 'foodseg_dataset', 'foodseg_dataset_v2', 'foodseg_two_class']:
        for split in ['train', 'validation', 'test']:
            path = BASE / name / f'{split}.txt'
            if path.exists():
                ids |= {Path(line).stem for line in path.read_text().split() if line.strip()}
    for probe in [PROBE_A, PROBE_B]:
        for path in (probe / 'images').glob('*.jpg'):
            ids.add(path.stem)
    return ids


def build_new_probe():
    (PROBE_B / 'images').mkdir(parents=True, exist_ok=True)
    used = used_image_ids()
    candidates = []
    for split in ['validation', 'test']:
        seen = {}
        with (RAW / f'oi_{split}.csv').open(encoding='utf-8', newline='') as handle:
            reader = csv.reader(handle)
            index = {name: i for i, name in enumerate(next(reader))}
            for row in reader:
                if row[index['LabelName']] not in MIDS:
                    continue
                image_id = row[index['ImageID']]
                if image_id in used:
                    continue
                seen[image_id] = split
        candidates += [{'split': split, 'image_id': image_id} for image_id, split in seen.items()]
    random.Random(1234).shuffle(candidates)
    chosen = candidates[:NEW_PROBE]
    done = []

    def fetch(record):
        dest = PROBE_B / 'images' / f"{record['image_id']}.jpg"
        url = f"https://open-images-dataset.s3.amazonaws.com/{record['split']}/{record['image_id']}.jpg"
        try:
            if not dest.exists():
                response = requests.get(url, timeout=60)
                response.raise_for_status()
                dest.write_bytes(response.content)
            with Image.open(dest) as image:
                image.verify()
            return {**record, 'url': url}
        except Exception as exc:  # noqa: BLE001
            return {**record, 'error': str(exc)}

    with ThreadPoolExecutor(max_workers=8) as pool:
        for future in as_completed([pool.submit(fetch, record) for record in chosen]):
            done.append(future.result())
    done = [record for record in done if 'error' not in record]
    attribution = []
    for split in ['validation', 'test']:
        wanted = {record['image_id'] for record in done if record['split'] == split}
        if not wanted:
            continue
        with (RAW / f'oi_meta_{split}.csv').open(encoding='utf-8', newline='') as handle:
            for row in csv.DictReader(handle):
                if row['ImageID'] in wanted:
                    attribution.append(row)
    if attribution:
        with (PROBE_B / 'attribution.csv').open('w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(attribution[0]))
            writer.writeheader()
            writer.writerows(attribution)
    (PROBE_B / 'manifest.json').write_text(json.dumps({
        'purpose': 'fresh kitchen probe; no model has been trained or tuned on these images',
        'images': len(done), 'source': 'Open Images validation+test, CC BY 2.0 (see attribution.csv)'},
        indent=2, ensure_ascii=False))
    return len(done)


def main():
    labels = json.loads((BASE / 'board_real_labels.json').read_text())
    labels2 = json.loads((BASE / 'board_real_labels_round2.json').read_text())
    negatives = labels['no_board'] + labels2['no_board']
    positives = {**labels['board_accept_box'], **labels2['board_accept_box']}
    probe_data = json.loads((PROBE_A / 'domain_probe.json').read_text())
    by_id = {record['image'].replace('.jpg', ''): record for record in probe_data['images']}

    for split in ['train', 'validation', 'test']:
        for path in (SRC / 'images' / split).glob('*.jpg'):
            link(path, OUT / 'images' / split / path.name)
        for path in (SRC / 'labels' / split).glob('*.txt'):
            link(path, OUT / 'labels' / split / path.name)
    if (SRC / 'attribution.csv').exists():
        link(SRC / 'attribution.csv', OUT / 'attribution.csv')

    def resolve(prefix):
        for full, record in by_id.items():
            if full.startswith(prefix):
                return full, record
        return None, None

    added = {'positives': 0, 'negatives': 0, 'missing': 0}
    for prefix in negatives:
        full, _ = resolve(prefix)
        source = PROBE_A / 'images' / f'{full}.jpg' if full else None
        if source is None or not source.exists():
            added['missing'] += 1
            continue
        link(source, OUT / 'images' / 'train' / f'{full}.jpg')
        (OUT / 'labels' / 'train' / f'{full}.txt').write_text('')
        added['negatives'] += 1
    for prefix, box_number in positives.items():
        full, record = resolve(prefix)
        source = PROBE_A / 'images' / f'{full}.jpg' if full else None
        if source is None or record is None or not source.exists():
            added['missing'] += 1
            continue
        x1, y1, x2, y2 = record['boards'][box_number - 1]['xyxy']
        with Image.open(source) as image:
            width, height = image.size
        link(source, OUT / 'images' / 'train' / f'{full}.jpg')
        (OUT / 'labels' / 'train' / f'{full}.txt').write_text(
            f'0 {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} '
            f'{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}\n')
        added['positives'] += 1

    lists = {}
    for split in ['train', 'validation', 'test']:
        paths = sorted((OUT / 'images' / split).glob('*.jpg'))
        (OUT / f'{split}.txt').write_text('\n'.join(p.as_posix() for p in paths) + '\n')
        lists[split] = len(paths)
    (OUT / 'data.yaml').write_text(
        f"path: {OUT.as_posix()}\ntrain: train.txt\nval: validation.txt\ntest: test.txt\nnames:\n  0: cutting_board\n")

    fresh = build_new_probe()
    (OUT / 'manifest.json').write_text(json.dumps({
        'parent': str(SRC), 'images': lists, 'added': added,
        'labels': ['board_real_labels.json', 'board_real_labels_round2.json'],
        'fresh_probe': {'path': str(PROBE_B), 'images': fresh,
                        'why': 'all previously probed kitchen images are now training data, so evaluation needs new ones'},
        'caveat': 'hard negatives are images where a cutting board was judged absent; the label is single-rater.',
    }, indent=2, ensure_ascii=False))
    print('BOARD_V3_READY', json.dumps({'splits': lists, 'added': added, 'fresh_probe': fresh}, ensure_ascii=False))


if __name__ == '__main__':
    main()
