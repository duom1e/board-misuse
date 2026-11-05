"""正例复核卡：整帧 + 四象限放大，用来先确认"这真的是板吗"再画框。

为什么需要：2026-11-04 发现被当作"圆板正例"的 BV1dHckzwEBA_11 其实是炒锅——
**正例标注本身也会错**。四象限（而不是只裁中心）是因为板常出现在画面边缘，
例如 BV1rzdZBTEps_180s 的红色案板就在左下角。

用法：
  python verify_board_positives.py                 # 扫 v1–v5 的 board_present=yes
  python verify_board_positives.py --max 8         # 只做前 8 张
输出：experiments/label_verify_v1/verify_sheet_*.jpg + queue.json
"""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parent
FONT = Path(r'C:\Windows\Fonts\arial.ttf')
EXPERIMENTS = ['new_source_video_v1', 'new_source_video_v2', 'new_source_video_v3',
               'new_source_video_v4', 'new_source_video_v5']
CARD_W = 600
CARD_H = 344
PER_SHEET = 3


def gather():
    items = []
    for name in EXPERIMENTS:
        labels_file = BASE / 'experiments' / name / 'labels_blind.json'
        if not labels_file.exists():
            continue
        labels = json.loads(labels_file.read_text(encoding='utf-8'))['labels']
        for frame_name, entry in labels.items():
            if entry.get('board_present') != 'yes' or entry.get('usable') is False:
                continue
            frame_path = BASE / 'experiments' / name / 'frames' / frame_name
            if frame_path.exists():
                items.append({'experiment': name, 'frame': frame_name, 'path': frame_path,
                              'revised': bool(entry.get('label_original_blind'))})
    return items


def card(item):
    image = Image.open(item['path']).convert('RGB')
    width, height = image.size
    canvas = Image.new('RGB', (CARD_W, CARD_H), 'white')
    font = ImageFont.truetype(str(FONT), 13)
    left = image.copy()
    left.thumbnail((CARD_W // 2 - 14, 300))
    canvas.paste(left, (5, 26))
    draw = ImageDraw.Draw(canvas)
    tag = ' [revised]' if item['revised'] else ''
    draw.text((6, 6), item['frame'] + tag, fill='black', font=font)
    draw.text((CARD_W // 2 + 8, 6), 'quadrant crops 2x', fill='black', font=font)
    tile_w = (CARD_W // 2 - 16) // 2
    tile_h = 145
    for index in range(4):
        col, row = index % 2, index // 2
        x1, y1 = int(width * col * 0.5), int(height * row * 0.5)
        x2, y2 = int(width * (col * 0.5 + 0.5)), int(height * (row * 0.5 + 0.5))
        piece = image.crop((x1, y1, x2, y2)).resize((tile_w, tile_h), Image.LANCZOS)
        canvas.paste(piece, (CARD_W // 2 + 5 + col * tile_w, 26 + row * (tile_h + 4)))
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='experiments/label_verify_v1')
    parser.add_argument('--max', type=int, default=0)
    args = parser.parse_args()
    out = BASE / args.out
    out.mkdir(parents=True, exist_ok=True)

    items = gather()
    if args.max:
        items = items[:args.max]
    for start in range(0, len(items), PER_SHEET):
        sheet = Image.new('RGB', (CARD_W, CARD_H * PER_SHEET), 'white')
        for slot, item in enumerate(items[start:start + PER_SHEET]):
            sheet.paste(card(item), (0, slot * CARD_H))
        sheet.save(out / ('verify_sheet_%d.jpg' % (start // PER_SHEET + 1)), quality=90)

    sheets = (len(items) + PER_SHEET - 1) // PER_SHEET
    (out / 'queue.json').write_text(json.dumps({
        'purpose': '核对"这真的是板吗"——正例标注同样会错（BV1dHckzwEBA_11 实为炒锅）',
        'layout': 'left: full frame; right: 2x2 quadrant crops (2x upscaled)',
        'total': len(items), 'sheets': sheets,
        'items': [{'experiment': i['experiment'], 'frame': i['frame'], 'revised': i['revised']} for i in items],
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'total': len(items), 'sheets': sheets}, ensure_ascii=False))


if __name__ == '__main__':
    main()
