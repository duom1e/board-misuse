"""Real-kitchen domain probe from Open Images (CC BY 2.0), no dish-only bias.

Why: every model so far was trained on dish close-ups plus a handful of board photos. This
builds a probe of real kitchen scenes (annotated with kitchen knife / kitchenware / cooking)
that are NOT used for training, then runs the board and food models on them so the domain gap
is visible instead of assumed away.

Probe images come only from the validation and test splits, which never feed training.
"""
import csv
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image

BASE = Path(__file__).resolve().parent
RAW = BASE.parent / 'work' / 'datasets'
OUT = BASE / 'real_kitchen_probe'
# Kitchen-ish annotations that a real kitchen scene usually carries.
MIDS = {'/m/058qzx': 'Kitchen knife', '/m/03_wxk': 'Kitchenware', '/m/03hlz0c': 'Kitchen utensil',
        '/m/0h99cwc': 'Kitchen appliance', '/m/02pdsw': 'Cutting board'}
EXCLUDE_DATASETS = ['board_dataset', 'food_taxonomy_v1', 'food_taxonomy_v2', 'food_taxonomy_v3',
                    'foodseg_dataset', 'foodseg_dataset_v2', 'foodseg_two_class']
CAP = 140


def known_ids():
    ids = set()
    for name in EXCLUDE_DATASETS:
        for split in ['train', 'validation', 'test']:
            path = BASE / name / f'{split}.txt'
            if path.exists():
                ids |= {Path(line).stem for line in path.read_text().split() if line.strip()}
    return ids


def collect():
    used = known_ids()
    candidates = []
    for split in ['validation', 'test']:
        seen = {}
        with (RAW / f'oi_{split}.csv').open(encoding='utf-8', newline='') as handle:
            reader = csv.reader(handle)
            idx = {name: i for i, name in enumerate(next(reader))}
            for row in reader:
                mid = row[idx['LabelName']]
                if mid not in MIDS:
                    continue
                image_id = row[idx['ImageID']]
                if image_id in used:
                    continue
                seen.setdefault(image_id, set()).add(MIDS[mid])
        for image_id, labels in seen.items():
            candidates.append({'split': split, 'image_id': image_id, 'labels': sorted(labels)})
    random.Random(42).shuffle(candidates)
    return candidates[:CAP]


def download(record):
    dest = OUT / 'images' / f"{record['image_id']}.jpg"
    url = f"https://open-images-dataset.s3.amazonaws.com/{record['split']}/{record['image_id']}.jpg"
    for attempt in range(3):
        try:
            if not dest.exists():
                response = requests.get(url, timeout=60)
                response.raise_for_status()
                dest.write_bytes(response.content)
            with Image.open(dest) as image:
                image.verify()
            return {**record, 'url': url, 'file': str(dest)}
        except Exception as exc:  # noqa: BLE001
            if attempt == 2:
                return {**record, 'url': url, 'error': str(exc)}


def main():
    (OUT / 'images').mkdir(parents=True, exist_ok=True)
    records = collect()
    print('candidates', len(records), flush=True)
    done = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(download, record) for record in records]
        for count, future in enumerate(as_completed(futures), 1):
            done.append(future.result())
            if count % 50 == 0:
                print('downloaded', count, flush=True)
    done = [r for r in done if 'error' not in r]
    attribution = []
    for split in ['validation', 'test']:
        wanted = {r['image_id'] for r in done if r['split'] == split}
        if not wanted:
            continue
        with (RAW / f'oi_meta_{split}.csv').open(encoding='utf-8', newline='') as handle:
            for row in csv.DictReader(handle):
                if row['ImageID'] in wanted:
                    attribution.append(row)
    if attribution:
        with (OUT / 'attribution.csv').open('w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(attribution[0]))
            writer.writeheader()
            writer.writerows(attribution)
    (OUT / 'manifest.json').write_text(json.dumps({
        'purpose': 'Real kitchen scenes (kitchen knife / kitchenware / cutting board annotations) used only as a '
                   'domain probe; never trained on.',
        'source': 'Open Images v7 validation+test splits, CC BY 2.0 per image (see attribution.csv)',
        'labels_of_interest': MIDS,
        'images': len(done),
        'excluded_from_training': EXCLUDE_DATASETS,
        'note': 'Annotation presence does not guarantee a cutting board is visible; the probe measures behaviour, '
                'not accuracy.',
    }, indent=2, ensure_ascii=False))
    print('PROBE_READY', len(done))


if __name__ == '__main__':
    main()
