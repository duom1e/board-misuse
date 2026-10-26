"""Screen Pexels candidates for real kitchen/prep scenes using slug rules + a COCO sanity check.

Step 1: strict slug rules (space word + people/action word, minus food-only words).
Step 2: download a larger sample.
Step 3: run generic yolo11n (COCO) and keep images that look like kitchens (sink/oven/fridge/
        microwave/dining table present, and not just a close-up of a dish).
Step 4: contact sheets for the survivors so a human (me) can confirm.

COCO is only a coarse content filter here; the final call is visual.
"""
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parent
SRC = BASE / 'pexels_kitchen'
OUT = SRC / 'images2'
FONT = Path(r'C:\Windows\Fonts\arial.ttf')
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126.0 Safari/537.36'}
SAMPLE = 320

SPACE_WORDS = ('kitchen', 'canteen', 'catering', 'restaurant', 'prep area')
PEOPLE_WORDS = ('chef', 'cook', 'staff', 'worker', 'woman', 'man', 'person', 'people', 'prepar', 'prep', 'serving')
FOOD_ONLY = ('dessert', 'cake', 'cookie', 'cupcake', 'doughnut', 'donut', 'pizza', 'burger', 'sushi',
             'coffee', 'cocktail', 'wine', 'beer', 'bread', 'pastry', 'ice cream', 'chocolate',
             'macaron', 'pancake', 'waffle', 'juice', 'smoothie')
KITCHEN_CLASSES = {'sink', 'oven', 'microwave', 'refrigerator', 'dining table', 'bowl', 'knife'}


def keep(slug):
    text = slug.lower()
    if any(word in text for word in FOOD_ONLY):
        return False
    return any(word in text for word in SPACE_WORDS) and any(word in text for word in PEOPLE_WORDS)


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
        return {**row, 'file': str(dest)}
    except Exception as exc:  # noqa: BLE001
        return {**row, 'error': str(exc)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = json.loads((SRC / 'candidates.json').read_text())
    pool = [row for row in data['candidates'] if keep(row['slug'])]
    print('slug-filtered candidates', len(pool), flush=True)
    rng = random.Random(777)
    rng.shuffle(pool)
    sample = sorted(pool[:SAMPLE], key=lambda r: r['id'])

    done = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        for count, future in enumerate(as_completed([executor.submit(download, row) for row in sample]), 1):
            done.append(future.result())
            if count % 80 == 0:
                print('downloaded', count, flush=True)
    ok = [row for row in done if 'error' not in row]
    print('downloaded ok', len(ok), flush=True)

    from ultralytics import YOLO
    model = YOLO('yolo11n.pt')
    scored = []
    for row in ok:
        result = model.predict(source=row['file'], conf=0.3, imgsz=960, device=0, verbose=False)[0]
        names = [model.names[int(c)] for c in result.boxes.cls.cpu().tolist()] if result.boxes is not None and len(result.boxes) else []
        kitchen_hits = sorted({name for name in names if name in KITCHEN_CLASSES})
        scored.append({**row, 'coco': sorted(set(names)), 'kitchen_hits': kitchen_hits,
                       'score': len(kitchen_hits) + (1 if 'person' in names else 0)})
    scored.sort(key=lambda row: (-row['score'], row['id']))
    survivors = [row for row in scored if len(row['kitchen_hits']) >= 2][:48]
    print('survivors', len(survivors), flush=True)

    font = ImageFont.truetype(str(FONT), 12)
    for index in range(0, len(survivors), 12):
        chunk = survivors[index:index + 12]
        canvas = Image.new('RGB', (960, 720), 'white')
        for slot, row in enumerate(chunk):
            image = Image.open(row['file']).convert('RGB')
            image.thumbnail((320, 160))
            canvas.paste(image, ((slot % 3) * 320, (slot // 3) * 180))
            ImageDraw.Draw(canvas).text(((slot % 3) * 320 + 3, (slot // 3) * 180 + 3),
                                        f"{row['id']} {len(row['kitchen_hits'])}",
                                        fill='yellow', font=font)
        canvas.save(SRC / f'survivors_{index // 12 + 1}.jpg')

    (SRC / 'screen_result.json').write_text(json.dumps({
        'slug_filtered': len(pool), 'sampled': len(sample), 'downloaded': len(ok),
        'coco_rule': 'keep if at least 2 of sink/oven/microwave/refrigerator/dining table/bowl/knife detected',
        'survivors': [{'id': row['id'], 'slug': row['slug'], 'page': row['page'],
                       'kitchen_hits': row['kitchen_hits'], 'coco': row['coco']} for row in survivors],
        'all_scored': [{'id': row['id'], 'score': row['score'], 'kitchen_hits': row['kitchen_hits']} for row in scored],
        'license': 'Pexels licence: free to use, attribution not required',
        'note': 'COCO is a coarse filter; final confirmation is visual on the survivor sheets.',
    }, indent=2, ensure_ascii=False))
    print(json.dumps({'slug_filtered': len(pool), 'downloaded': len(ok), 'survivors': len(survivors)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
