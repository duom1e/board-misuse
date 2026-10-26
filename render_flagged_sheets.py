"""Render every flagged Pexels image into sheets for verification (not just the first 24)."""
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parent
PROBE = BASE / 'pexels_kitchen' / 'probe'
FONT = Path(r'C:\Windows\Fonts\arial.ttf')


def main():
    data = json.loads((PROBE / 'summary.json').read_text())
    flagged = [record for record in data['records'] if record['boards'] or record['foods']]
    font = ImageFont.truetype(str(FONT), 12)
    index = []
    for start in range(0, len(flagged), 12):
        canvas = Image.new('RGB', (960, 720), 'white')
        for slot, record in enumerate(flagged[start:start + 12]):
            image = Image.open(record['image']).convert('RGB')
            draw = ImageDraw.Draw(image)
            for box in record['boards']:
                draw.rectangle(box['xyxy'], outline='blue', width=3)
            for box in record['foods']:
                draw.rectangle(box['xyxy'], outline='red', width=3)
            image.thumbnail((320, 160))
            canvas.paste(image, ((slot % 3) * 320, (slot // 3) * 180))
            ImageDraw.Draw(canvas).text(((slot % 3) * 320 + 3, (slot // 3) * 180 + 3),
                                        record['name'].replace('pexels_', '').replace('.jpg', ''),
                                        fill='yellow', font=font)
            index.append({'sheet': start // 12 + 1, 'slot': slot + 1, 'name': record['name'],
                          'board_boxes': len(record['boards']), 'food_boxes': len(record['foods'])})
        canvas.save(PROBE / f'all_flagged_{start // 12 + 1}.jpg')
    (PROBE / 'all_flagged_index.json').write_text(json.dumps(index, ensure_ascii=False, indent=1))
    print('sheets', (len(flagged) + 11) // 12, 'flagged', len(flagged))


if __name__ == '__main__':
    main()
