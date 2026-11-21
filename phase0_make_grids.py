"""阶段 0-e：为每段有板来源生成 5×4 放大对照图（每格 300×170），用于逐帧判遮挡档位。

缩略图（245×130）不足以判 mild/severe，本脚本按来源单独出大图。
"""
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parent
OUT = BASE / 'experiments' / 'occlusion_phase0'
BOARD_SOURCES = ['BV1684y1r7HX', 'BV1j441157px', 'BV1ux411g7tW',
                 'BV1nm4y1Z7qs', 'BV1wS4y1T78o', 'BV1dh41187N8']
TILE = (300, 170)


def main():
    manifest = json.loads((OUT / 'blind_v2_manifest.json').read_text(encoding='utf-8'))
    font = ImageFont.truetype(r'C:\Windows\Fonts\arial.ttf', 13)
    for bvid in BOARD_SOURCES:
        frames = [f for f in manifest['frames'] if f['source_bvid'] == bvid]
        if not frames:
            print('missing', bvid)
            continue
        canvas = Image.new('RGB', (TILE[0] * 5, (TILE[1] + 18) * 4), 'white')
        for slot, record in enumerate(frames[:20]):
            image = Image.open(record['file']).convert('RGB')
            image = image.resize(TILE, Image.LANCZOS)
            x = (slot % 5) * TILE[0]
            y = (slot // 5) * (TILE[1] + 18)
            canvas.paste(image, (x, y))
            ImageDraw.Draw(canvas).text((x + 2, y + TILE[1] + 1),
                                        '#%d %.0fs' % (slot, record['seconds']), fill='black', font=font)
        path = OUT / ('grid_%s.jpg' % bvid)
        canvas.save(path, quality=90)
        print(bvid, len(frames), '->', path.name)


if __name__ == '__main__':
    main()
