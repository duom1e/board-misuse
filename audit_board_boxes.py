"""按《砧板定义》回查已标注的人工框：把框画回原图，检查是不是"真板"。

风险点来自 v7：编织竹席曾被当成木砧板。本脚本优先检查同一段视频（BV1bqsfzbELh）里
的已标帧，因为那里既有木砧板也有竹席。

用法：python audit_board_boxes.py [--experiment new_source_video_v4] [--per-sheet 4]
"""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parent
FONT = Path(r'C:\Windows\Fonts\arial.ttf')
CARD_W, CARD_H = 620, 300


def collect(experiment):
    directory = BASE / 'experiments' / experiment / 'frames'
    items = []
    for sidecar in sorted(directory.glob('*.board.json')):
        payload = json.loads(sidecar.read_text(encoding='utf-8'))
        image = Path(payload['image'])
        if image.exists():
            items.append({'frame': image.name, 'image': image, 'box': payload['board']})
    return items


def card(item):
    image = Image.open(item['image']).convert('RGB')
    width, height = image.size
    canvas = Image.new('RGB', (CARD_W, CARD_H), 'white')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(str(FONT), 13)
    full = image.copy()
    full.thumbnail((CARD_W // 2 - 14, CARD_H - 26))
    draw.rectangle((5, 20, 5 + full.width, 20 + full.height), outline='gray')
    canvas.paste(full, (5, 20))
    bx = [item['box'][0] * full.width / 1000, item['box'][1] * full.height / 1000,
          item['box'][2] * full.width / 1000, item['box'][3] * full.height / 1000]
    draw.rectangle((5 + bx[0], 20 + bx[1], 5 + bx[2], 20 + bx[3]), outline='lime', width=3)
    # 框内裁切放大，专门看材质
    x1 = int(image.width * item['box'][0] / 1000)
    y1 = int(image.height * item['box'][1] / 1000)
    x2 = int(image.width * item['box'][2] / 1000)
    y2 = int(image.height * item['box'][3] / 1000)
    if x2 - x1 > 8 and y2 - y1 > 8:
        crop = image.crop((x1, y1, x2, y2)).resize((CARD_W // 2 - 14, CARD_H - 26), Image.LANCZOS)
        canvas.paste(crop, (CARD_W // 2 + 5, 20))
    draw.text((6, 3), item['frame'], fill='black', font=font)
    draw.text((CARD_W // 2 + 8, 3), 'inside box (2x)', fill='black', font=font)
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', default='new_source_video_v4')
    parser.add_argument('--per-sheet', type=int, default=4)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    out = BASE / 'experiments' / 'board_box_audit_v1' / args.experiment
    out.mkdir(parents=True, exist_ok=True)

    items = collect(args.experiment)
    for start in range(0, len(items), args.per_sheet):
        sheet = Image.new('RGB', (CARD_W, CARD_H * args.per_sheet), 'white')
        for slot, item in enumerate(items[start:start + args.per_sheet]):
            sheet.paste(card(item), (0, slot * CARD_H))
        sheet.save(out / ('audit_%d.jpg' % (start // args.per_sheet + 1)), quality=90)
    (out / 'items.json').write_text(json.dumps(
        [{'frame': i['frame'], 'box': i['box']} for i in items], ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'experiment': args.experiment, 'frames': len(items),
                      'sheets': (len(items) + args.per_sheet - 1) // args.per_sheet}, ensure_ascii=False))


if __name__ == '__main__':
    main()
