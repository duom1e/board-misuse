"""批量候选框预览（第二轮标注）：8 帧候选，一次出图核对。"""
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parent
OUT = BASE / 'experiments' / 'board_box_preview_v2'
FONT = Path(r'C:\Windows\Fonts\arial.ttf')
CANDIDATES = [
    ('new_source_video_v5', 'BV1pu411U7Zh_115s.jpg', [250, 200, 1000, 900]),
    ('new_source_video_v5', 'BV1pu411U7Zh_140s.jpg', [350, 200, 1000, 1000]),
    ('new_source_video_v5', 'BV1pu411U7Zh_165s.jpg', [150, 300, 800, 1000]),
    ('new_source_video_v5', 'BV18pRhBgEyw_870s.jpg', [760, 60, 1000, 280]),
    ('new_source_video_v4', 'BV1bqsfzbELh_300s.jpg', [0, 30, 1000, 600]),
    ('new_source_video_v4', 'BV1bqsfzbELh_440s.jpg', [100, 250, 900, 560]),
    ('new_source_video_v3', 'BV12a41187Ja_240s.jpg', [350, 600, 1000, 1000]),
    ('new_source_video_v3', 'BV1pq4y147jk_290s.jpg', [0, 200, 1000, 800]),
]
TILE_W, TILE_H = 480, 270


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype(str(FONT), 14)
    canvas = Image.new('RGB', (TILE_W * 2, TILE_H * 4), 'white')
    for index, (folder, name, box) in enumerate(CANDIDATES):
        path = BASE / 'experiments' / folder / 'frames' / name
        image = Image.open(path).convert('RGB').resize((TILE_W, TILE_H))
        draw = ImageDraw.Draw(image)
        pixels = [box[0] * TILE_W / 1000, box[1] * TILE_H / 1000,
                  box[2] * TILE_W / 1000, box[3] * TILE_H / 1000]
        draw.rectangle(pixels, outline='lime', width=3)
        draw.text((max(2, pixels[0]) + 2, max(0, pixels[1] - 16)), '#%d %s' % (index, box),
                  fill='lime', font=font)
        canvas.paste(image, ((index % 2) * TILE_W, (index // 2) * TILE_H))
        ImageDraw.Draw(canvas).text(((index % 2) * TILE_W + 4, (index // 2) * TILE_H + TILE_H - 16),
                                    name, fill='yellow', font=font)
    canvas.save(OUT / 'preview.jpg', quality=90)
    (OUT / 'candidates.json').write_text(json.dumps(
        [{'folder': f, 'frame': n, 'box': b} for f, n, b in CANDIDATES], ensure_ascii=False, indent=2),
        encoding='utf-8')
    print(str(OUT / 'preview.jpg'))


if __name__ == '__main__':
    main()
