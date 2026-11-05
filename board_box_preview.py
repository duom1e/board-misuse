"""按给定归一化框批量生成预览对照表（人工画框的复核步骤）。

用法：python board_box_preview.py          # 用内置候选框生成 <out>/preview.jpg
"""
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parent
OUT = BASE / 'experiments' / 'board_box_preview_v1'
FONT = Path(r'C:\Windows\Fonts\arial.ttf')
CANDIDATES = [
    ('new_source_video_v3', 'BV1pq4y147jk_050s.jpg', [0, 120, 1000, 1000]),
    ('new_source_video_v3', 'BV12a41187Ja_090s.jpg', [0, 0, 900, 1000]),
    ('new_source_video_v4', 'BV1bqsfzbELh_125s.jpg', [0, 560, 700, 1000]),
    ('new_source_video_v4', 'BV1bqsfzbELh_475s.jpg', [0, 300, 1000, 1000]),
]
TILE_W, TILE_H = 480, 270


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype(str(FONT), 14)
    canvas = Image.new('RGB', (TILE_W * 2, TILE_H * 2), 'white')
    for index, (folder, name, box) in enumerate(CANDIDATES):
        path = BASE / 'experiments' / folder / 'frames' / name
        image = Image.open(path).convert('RGB').resize((TILE_W, TILE_H))
        draw = ImageDraw.Draw(image)
        pixels = [box[0] * TILE_W / 1000, box[1] * TILE_H / 1000, box[2] * TILE_W / 1000, box[3] * TILE_H / 1000]
        draw.rectangle(pixels, outline='lime', width=4)
        draw.text((pixels[0] + 4, max(0, pixels[1] - 16)), f"#{index} {box}", fill='lime', font=font)
        canvas.paste(image, ((index % 2) * TILE_W, (index // 2) * TILE_H))
        ImageDraw.Draw(canvas).text(((index % 2) * TILE_W + 4, (index // 2) * TILE_H + TILE_H - 18),
                                    name, fill='yellow', font=font)
    canvas.save(OUT / 'preview.jpg', quality=92)
    (OUT / 'candidates.json').write_text(json.dumps(
        [{'folder': f, 'frame': n, 'box': b} for f, n, b in CANDIDATES], ensure_ascii=False, indent=2), encoding='utf-8')
    print(str(OUT / 'preview.jpg'))


if __name__ == '__main__':
    main()
