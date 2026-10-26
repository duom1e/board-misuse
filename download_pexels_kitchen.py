"""Download a sampled set of Pexels kitchen photos and build contact sheets."""
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parent
SRC = BASE / 'pexels_kitchen'
OUT = SRC / 'images'
FONT = Path(r'C:\Windows\Fonts\arial.ttf')
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126.0 Safari/537.36'}
PER_GROUP = 40

GROUPS = {
    'commercial': ['commercial', 'restaurant kitchen', 'canteen', 'catering', 'professional kitchen', 'industrial kitchen'],
    'chef': ['chef', 'cook'],
    'kitchen': ['kitchen'],
}


def classify(slug):
    text = slug.lower()
    if any(word in text for word in ('commercial', 'canteen', 'catering', 'industrial', 'restaurant')):
        return 'commercial'
    if 'chef' in text or 'cook' in text:
        return 'chef'
    if 'kitchen' in text:
        return 'kitchen'
    return None


def pick():
    data = json.loads((SRC / 'candidates.json').read_text())
    buckets = {name: [] for name in GROUPS}
    for row in data['candidates']:
        group = classify(row['slug'])
        if group:
            buckets[group].append(row)
    rng = random.Random(20261026)
    chosen = []
    for name, rows in buckets.items():
        rows = sorted(rows, key=lambda r: r['id'])
        rng.shuffle(rows)
        chosen += [{**row, 'group': name} for row in rows[:PER_GROUP]]
        print(name, 'available', len(rows), 'picked', min(PER_GROUP, len(rows)), flush=True)
    return chosen


def download(row):
    dest = OUT / f"pexels_{row['id']}.jpg"
    url = f"https://images.pexels.com/photos/{row['id']}/pexels-photo-{row['id']}.jpeg?auto=compress&cs=tinysrgb&w=1200"
    try:
        if not dest.exists():
            response = requests.get(url, headers=HEADERS, timeout=60)
            response.raise_for_status()
            dest.write_bytes(response.content)
        with Image.open(dest) as image:
            image.verify()
        return {**row, 'url': url, 'bytes': dest.stat().st_size}
    except Exception as exc:  # noqa: BLE001
        return {**row, 'error': str(exc)}


def sheet(records, name, per_sheet=12):
    font = ImageFont.truetype(str(FONT), 12)
    for index in range(0, len(records), per_sheet):
        chunk = records[index:index + per_sheet]
        canvas = Image.new('RGB', (960, 720), 'white')
        for slot, record in enumerate(chunk):
            path = OUT / f"pexels_{record['id']}.jpg"
            try:
                image = Image.open(path).convert('RGB')
            except Exception:  # noqa: BLE001
                continue
            image.thumbnail((320, 160))
            canvas.paste(image, ((slot % 3) * 320, (slot // 3) * 180))
            ImageDraw.Draw(canvas).text(((slot % 3) * 320 + 3, (slot // 3) * 180 + 3),
                                        f"{record['id']}", fill='yellow', font=font)
        canvas.save(SRC / f'{name}_{index // per_sheet + 1}.jpg')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = pick()
    done = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for count, future in enumerate(as_completed([pool.submit(download, row) for row in rows]), 1):
            done.append(future.result())
            if count % 40 == 0:
                print('downloaded', count, flush=True)
    ok = [row for row in done if 'error' not in row]
    for group in GROUPS:
        sheet([row for row in ok if row['group'] == group], f'sheet_{group}')
    (SRC / 'manifest.json').write_text(json.dumps({
        'purpose': 'Pexels kitchen photos for a real-kitchen probe; not a canteen dataset',
        'source': 'Pexels photo sitemaps, images from images.pexels.com',
        'license': 'Pexels licence: free to use, attribution not required',
        'groups': GROUPS, 'requested': len(rows), 'downloaded': len(ok),
        'failed': [row for row in done if 'error' in row],
        'photos': ok,
        'note': 'Slug keywords decide the group; actual content must be checked visually.',
    }, indent=2, ensure_ascii=False))
    print(json.dumps({'ok': len(ok), 'by_group': {g: sum(1 for r in ok if r['group'] == g) for g in GROUPS}},
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
