"""第三批候选框预览：8 帧，先看再落盘。"""
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parent
OUT = BASE / 'experiments' / 'board_box_preview_v3'
FONT = Path(r'C:\Windows\Fonts\arial.ttf')
CANDIDATES = [
    ('new_source_video_v1', 'BV1rzdZBTEps_180s.jpg', [0, 600, 620, 1000]),
    ('new_source_video_v2', 'BV1ocXNBrE66_010s.jpg', [0, 100, 1000, 1000]),
    ('new_source_video_v2', 'BV1ocXNBrE66_190s.jpg', [550, 600, 1000, 1000]),
    ('new_source_video_v3', 'BV12a41187Ja_190s.jpg', [250, 300, 1000, 900]),
    ('new_source_video_v3', 'BV1pq4y147jk_020s.jpg', [0, 150, 1000, 1000]),
    ('new_source_video_v3', 'BV1pq4y147jk_200s.jpg', [0, 100, 1000, 900]),
    ('new_source_video_v4', 'BV1bqsfzbELh_055s.jpg', [0, 550, 360, 1000]),
    ('new_source_video_v4', 'BV1bqsfzbELh_230s.jpg', [0, 480, 1000, 1000]),
]
W, H = 480, 270


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype(str(FONT), 14)
    canvas = Image.new('RGB', (W * 2, H * 4), 'white')
    for i, (folder, name, box) in enumerate(CANDIDATES):
        im = Image.open(BASE / 'experiments' / folder / 'frames' / name).convert('RGB').resize((W, H))
        d = ImageDraw.Draw(im)
        px = [box[0] * W / 1000, box[1] * H / 1000, box[2] * W / 1000, box[3] * H / 1000]
        d.rectangle(px, outline='lime', width=3)
        d.text((max(2, px[0]) + 2, max(0, px[1] - 16)), '#%d %s' % (i, box), fill='lime', font=font)
        canvas.paste(im, ((i % 2) * W, (i // 2) * H))
        ImageDraw.Draw(canvas).text(((i % 2) * W + 4, (i // 2) * H + H - 16), name, fill='yellow', font=font)
    canvas.save(OUT / 'preview.jpg', quality=90)
    (OUT / 'candidates.json').write_text(
        json.dumps([{'folder': f, 'frame': n, 'box': b} for f, n, b in CANDIDATES], ensure_ascii=False, indent=2),
        encoding='utf-8')
    print(OUT / 'preview.jpg')


if __name__ == '__main__':
    main()
