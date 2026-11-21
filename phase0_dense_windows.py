"""阶段 0-g：为"错误继承"（false persistence）构造**密集采样窗口**。

需求：时序规则最容易犯的错是"板已经离开画面，却还在报"。
要测这个，必须有"板存在 → 板消失"的连续帧，而不是每隔 4–5 秒抽一帧。

做法：在已知发生"板离开"的视频段落附近，按 2 fps 密集抽帧，
窗口 = 切换点前 15 秒 + 后 10 秒，共约 50 帧；出 5×10 大图供确认切换时刻。
"""
import json
from pathlib import Path

import cv2

BASE = Path(__file__).resolve().parent
INBOX = BASE.parent / '待导入视频'
OUT = BASE / 'experiments' / 'occlusion_phase0'
WINDOWS = [
    ('BV1ux411g7tW', 205.0, 25.0, '主厨刀工段 → 切到采访（板离开画面）'),
    ('BV1wS4y1T78o', 74.0, 16.0, '鱼生切完 → 装盆（板离开画面）'),
    ('BV1dh41187N8', 424.0, 20.0, '刀工演示 → 手持成品特写（板基本出画）'),
]
FPS_SAMPLE = 2.0


def main():
    records = []
    for bvid, start, length, note in WINDOWS:
        video = next(INBOX.glob(f'*{bvid}*.mp4'))
        capture = cv2.VideoCapture(str(video))
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        count = int(length * FPS_SAMPLE)
        for k in range(count):
            seconds = round(start + k / FPS_SAMPLE, 2)
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(seconds * fps))
            ok, frame = capture.read()
            if not ok:
                continue
            name = 'w_%s_%07.2fs.jpg' % (bvid, seconds)
            path = OUT / 'frames' / name
            cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            records.append({'source_bvid': bvid, 'seconds': seconds, 'file': str(path),
                            'window_note': note})
        capture.release()
        print(bvid, 'start', start, 'frames', len([r for r in records if r['source_bvid'] == bvid]))

    (OUT / 'persistence_windows.json').write_text(json.dumps({
        'plan': 'false_persistence_v1',
        'why': '错误继承需要"板存在→板消失"的连续帧，2 fps 密集采样',
        'windows': [{'bvid': b, 'start': s, 'length': l, 'note': n} for b, s, l, n in WINDOWS],
        'frames': records,
        'model_inference_performed': False,
    }, ensure_ascii=False, indent=2), encoding='utf-8')

    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype(r'C:\Windows\Fonts\arial.ttf', 11)
    for bvid, start, length, note in WINDOWS:
        frames = [r for r in records if r['source_bvid'] == bvid]
        if not frames:
            continue
        cols = 10
        rows = (len(frames) + cols - 1) // cols
        canvas = Image.new('RGB', (180 * cols, (120 + 14) * rows), 'white')
        for slot, record in enumerate(frames):
            image = Image.open(record['file']).convert('RGB').resize((180, 120), Image.LANCZOS)
            x, y = (slot % cols) * 180, (slot // cols) * 134
            canvas.paste(image, (x, y))
            ImageDraw.Draw(canvas).text((x + 2, y + 121), '%.1fs' % record['seconds'],
                                        fill='black', font=font)
        canvas.save(OUT / ('window_%s.jpg' % bvid), quality=88)
        print('grid', bvid, len(frames))


if __name__ == '__main__':
    main()
