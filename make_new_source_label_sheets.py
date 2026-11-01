"""为冻结的新来源帧生成两张对照表，供人先标注（此时模型尚未看过这些帧）。"""
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parent
OUT = BASE / 'experiments' / 'new_source_video_v1'
FONT = Path(r'C:\Windows\Fonts\arial.ttf')


def main():
    manifest = json.loads((OUT / 'frames_manifest.json').read_text(encoding='utf-8'))
    frames = manifest['frames']
    font = ImageFont.truetype(str(FONT), 13)
    for start in range(0, len(frames), 12):
        canvas = Image.new('RGB', (960, 720), 'white')
        for slot, record in enumerate(frames[start:start + 12]):
            image = Image.open(record['file']).convert('RGB')
            image.thumbnail((320, 160))
            canvas.paste(image, ((slot % 3) * 320, (slot // 3) * 180))
            ImageDraw.Draw(canvas).text(((slot % 3) * 320 + 3, (slot // 3) * 180 + 3),
                                        f"{record['source_bvid'][-4:]} {record['seconds']}s",
                                        fill='yellow', font=font)
        canvas.save(OUT / f'label_sheet_{start // 12 + 1}.jpg')
    print('sheets', (len(frames) + 11) // 12)


if __name__ == '__main__':
    main()
