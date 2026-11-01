"""给单张图设一个人工板框，并生成带框预览，供复核人确认。

用法：
  python set_board_box.py <图片> --box 120,180,860,720 [--out 同名.board.json]

输出 JSON：{"board": [x1,y1,x2,y2], "image": "...", "image_sha256": "...", "normalized": true}
被测图会同时生成 <图片>.board_preview.jpg，确认框位置无误后再交给 review_combined.py。
"""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parent
FONT = Path(r'C:\Windows\Fonts\arial.ttf')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path)
    parser.add_argument('--box', required=True, help='归一化 x1,y1,x2,y2（0-1000）')
    parser.add_argument('--out', type=Path, default=None)
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error('图片不存在')
    box = [float(v) for v in args.box.split(',')]
    if len(box) != 4 or not (0 <= box[0] < box[2] <= 1000 and 0 <= box[1] < box[3] <= 1000):
        parser.error('需要 4 个 0-1000 的数，且 x1<x2、y1<y2')

    image = Image.open(args.image).convert('RGB')
    width, height = image.size
    pixels = [box[0] * width / 1000, box[1] * height / 1000, box[2] * width / 1000, box[3] * height / 1000]
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(str(FONT), 20)
    draw.rectangle(pixels, outline='lime', width=5)
    draw.text((pixels[0] + 6, pixels[1] + 6), 'human board', fill='lime', font=font)
    preview = args.image.with_suffix('.board_preview.jpg')
    image.save(preview, quality=92)

    payload = {
        'board': box,
        'normalized': True,
        'image': str(args.image.resolve()),
        'image_sha256': hashlib.sha256(args.image.read_bytes()).hexdigest(),
        'source': 'human',
        'note': '人工框定板面；review_combined.py --board-json 可读取本文件',
    }
    target = args.out or args.image.with_suffix('.board.json')
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(str(target))
    print(str(preview))


if __name__ == '__main__':
    main()
