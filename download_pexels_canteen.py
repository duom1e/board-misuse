"""Download the 40 Pexels photos whose slug contains 'canteen' and build contact sheets."""
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parent
SRC = BASE / 'pexels_kitchen'
OUT = SRC / 'canteen'
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126.0 Safari/537.36'}


def download(row):
    dest = OUT / f"{row['id']}.jpg"
    url = f"https://images.pexels.com/photos/{row['id']}/pexels-photo-{row['id']}.jpeg?auto=compress&cs=tinysrgb&w=1200"
    try:
        if not dest.exists():
            response = requests.get(url, headers=HEADERS, timeout=60)
            response.raise_for_status()
            dest.write_bytes(response.content)
        with Image.open(dest) as image:
            image.verify()
        return row
    except Exception:  # noqa: BLE001
        return None


def main():
    OUT.mkdir(exist_ok=True)
    candidates = json.loads((SRC / 'candidates.json').read_text())['candidates']
    hits = [row for row in candidates if 'canteen' in row['slug'].lower()]
    ok = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for future in as_completed([pool.submit(download, row) for row in hits]):
            value = future.result()
            if value:
                ok.append(value)
    ok.sort(key=lambda row: row['id'])
    font = ImageFont.truetype(r'C:\Windows\Fonts\arial.ttf', 12)
    for start in range(0, len(ok), 12):
        canvas = Image.new('RGB', (960, 720), 'white')
        for slot, row in enumerate(ok[start:start + 12]):
            image = Image.open(OUT / f"{row['id']}.jpg").convert('RGB')
            image.thumbnail((320, 160))
            canvas.paste(image, ((slot % 3) * 320, (slot // 3) * 180))
            ImageDraw.Draw(canvas).text(((slot % 3) * 320 + 3, (slot // 3) * 180 + 3),
                                        str(row['id']), fill='yellow', font=font)
        canvas.save(OUT / f"sheet_{start // 12 + 1}.jpg")
    (OUT / 'manifest.json').write_text(json.dumps({
        'source': 'Pexels, slug contains canteen', 'license': 'Pexels licence: free to use',
        'downloaded': len(ok), 'of': len(hits), 'photos': ok}, ensure_ascii=False, indent=1))
    print('downloaded', len(ok), 'of', len(hits), 'sheets', (len(ok) + 11) // 12)


if __name__ == '__main__':
    main()
