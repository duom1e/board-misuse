"""Render numbered board proposals for visual accept/reject labelling.

Each flagged kitchen-probe image gets its own tile with the board model's boxes drawn and
indexed (b1, b2, ...). Verdicts are recorded by hand in real_board_labels.json; only accepted
boxes become training labels, and images with no board at all become hard negatives.
"""
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parent
PROBE = BASE / 'real_kitchen_probe'
FONT = Path(r'C:\Windows\Fonts\arial.ttf')
PER_SHEET = 12


def main():
    data = json.loads((PROBE / 'domain_probe.json').read_text())
    flagged = [r for r in data['images'] if r['boards']]
    font = ImageFont.truetype(str(FONT), 13)
    index = []
    for sheet_number, start in enumerate(range(0, len(flagged), PER_SHEET), 1):
        sheet = Image.new('RGB', (960, 720), 'white')
        for slot, record in enumerate(flagged[start:start + PER_SHEET]):
            image = Image.open(PROBE / 'images' / record['image']).convert('RGB')
            draw = ImageDraw.Draw(image)
            for number, box in enumerate(record['boards'], 1):
                draw.rectangle(box['xyxy'], outline='blue', width=3)
                draw.text((box['xyxy'][0] + 3, box['xyxy'][1] + 3), f"b{number} {box['confidence']:.2f}",
                          fill='blue', font=font)
            image.thumbnail((320, 160))
            sheet.paste(image, ((slot % 3) * 320, (slot // 3) * 180))
            index.append({'sheet': sheet_number, 'slot': slot + 1, 'image': record['image'],
                          'proposals': [round(b['confidence'], 3) for b in record['boards']]})
        sheet.save(PROBE / f'proposal_sheet_{sheet_number}.jpg')
    (PROBE / 'proposal_index.json').write_text(json.dumps(index, indent=2, ensure_ascii=False))
    print('sheets', (len(flagged) + PER_SHEET - 1) // PER_SHEET, 'images', len(flagged))


if __name__ == '__main__':
    main()
