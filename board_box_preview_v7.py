"""第四批候选框预览：4 个不同来源/场景的帧，先看再落盘。"""
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parent
OUT = BASE / 'experiments' / 'board_box_preview_v7'
FONT = Path(r'C:\Windows\Fonts\arial.ttf')
CANDIDATES = [
    ('new_source_video_v8', 'BV14W41197Rd_060s.jpg', [250, 300, 800, 950]),
    ('new_source_video_v8', 'BV14W41197Rd_180s.jpg', [300, 100, 950, 800]),
]
W, H = 480, 270


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype(str(FONT), 14)
    canvas = Image.new('RGB', (W * 2, H * 2), 'white')
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
